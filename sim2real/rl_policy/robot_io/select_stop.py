"""Latched Select software stop. SDK2 DDS runs outside the inline SDK process.

Not a hardware stop or a hard real-time watchdog. Select is Unitree key bit 3.
"""
from __future__ import annotations
import multiprocessing as mp
import threading
import time
import numpy as np
from loguru import logger
from sim2real.rl_policy.robot_io.base import RobotIO

SELECT_MASK = 1 << 3


def _receive_keys(keys, pressed, seen, received):
    if int(keys) & SELECT_MASK:
        pressed.set()  # Release packets cannot clear the latch.
    received.value = time.monotonic()
    seen.set()


def _receive_lowstate(msg, pressed, seen, received):
    remote = msg.wireless_remote
    if len(remote) < 4:
        return
    # Unitree remote packet: two header bytes, then little-endian key word.
    keys = int(remote[2]) | (int(remote[3]) << 8)
    _receive_keys(keys, pressed, seen, received)


def _listen(interface, domain_id, pressed, seen, received, stop, failed):
    subscribers = []
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        ChannelFactoryInitialize(domain_id, interface)
        subscriber = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
        subscribers.append(subscriber)
        subscriber.Init(lambda msg: _receive_keys(msg.keys, pressed, seen, received), 10)
        lowstate = ChannelSubscriber("rt/lowstate", LowState_)
        subscribers.append(lowstate)
        lowstate.Init(lambda msg: _receive_lowstate(msg, pressed, seen, received), 10)
        while not stop.wait(0.05):
            pass
    except BaseException:
        failed.set()
        raise
    finally:
        for subscriber in subscribers:
            subscriber.Close()


class SelectStopMonitor:
    """Read-only DDS listener; no robot mode switching or motor writes."""
    def __init__(self, interface, domain_id):
        self._listener_warning_logged = False
        context = mp.get_context("spawn")
        self.pressed = context.Event()
        self.seen = context.Event()
        self.received = context.Value("d", 0.0, lock=False)
        self.stop = context.Event()
        self.failed = context.Event()
        self.process = context.Process(
            target=_listen,
            args=(interface, domain_id, self.pressed, self.seen, self.received, self.stop, self.failed),
            name="g1-select-stop", daemon=True,
        )
        self.process.start()

    @property
    def reason(self):
        if self.pressed.is_set():
            return "Unitree remote Select pressed"
        if self.failed.is_set() or not self.process.is_alive():
            if not self._listener_warning_logged:
                logger.warning("Select listener unavailable; remote software stop cannot receive new presses")
                self._listener_warning_logged = True
        # An idle remote is not a stop request. Only an actual Select latches.
        return None

    def close(self):
        self.stop.set()
        self.process.join(timeout=1.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)


class SelectStopRobotIO(RobotIO):
    """Serialize motor writes and replace ALL later commands with damping.

    A thread can submit damping during inference if Python/SDK scheduling
    continues. kp=0, dq_target=0, tau_ff=0 leaves -kd * measured_velocity.
    """
    def __init__(self, backend, monitor, joint_count, *, kd=2.0):
        if not np.isfinite(kd) or kd <= 0:
            raise ValueError("Stop damping kd must be finite and positive")
        self.backend = backend
        self.monitor = monitor
        self._zeros = np.zeros(joint_count, dtype=np.float32)
        self._kd = np.full(joint_count, kd, dtype=np.float32)
        self._lock = threading.Lock()
        self._reason = None
        self._closed = False
        self._stop = threading.Event()
        self._thread = None
        self._write_error_logged = False

    def _poll_locked(self):
        if self._reason is None:
            reason = self.monitor.reason
            if reason is not None:
                self._reason = reason
                logger.error("SELECT SOFTWARE STOP LATCHED: {}. Damping only; restart required.", reason)
        return self._reason

    @property
    def emergency_stop_reason(self):
        with self._lock:
            return self._poll_locked()

    def _damp_locked(self):
        self.backend.write_command(self._zeros, self._zeros, self._zeros, self._zeros, self._kd)

    def poll_stop(self):
        with self._lock:
            if not self._closed and self._poll_locked() is not None:
                self._damp_locked()

    def start(self):
        self.poll_stop()
        def watch():
            while not self._stop.wait(0.01):
                try:
                    self.poll_stop()
                except Exception:
                    if not self._write_error_logged:
                        logger.exception("Failed to send Select stop damping command")
                        self._write_error_logged = True
        self._thread = threading.Thread(target=watch, name="g1-stop-writer", daemon=True)
        self._thread.start()

    def read_state(self):
        return self.backend.read_state()

    def write_command(self, q_target, dq_target, tau_ff, kp, kd):
        with self._lock:
            if self._closed:
                raise RuntimeError("SelectStopRobotIO is closed")
            if self._poll_locked() is not None:
                self._damp_locked()
            else:
                self.backend.write_command(q_target, dq_target, tau_ff, kp, kd)
                # Select may arrive during a motor write.
                if self._poll_locked() is not None:
                    self._damp_locked()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with self._lock:
            if self._closed:
                return
            try:
                if self._poll_locked() is not None:
                    self._damp_locked()
            finally:
                self._closed = True
                self.backend.close()
