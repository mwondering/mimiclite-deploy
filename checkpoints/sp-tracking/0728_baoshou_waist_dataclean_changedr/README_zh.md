# SP_Tracking SPV5-2 部署文件

源训练：`2026-07-28_05-25-04_SPTracking-G1-BFM-SPV5-2Actor-HEFTCritic-HEFTReward_6gpu_12288env_motion_data_correct`
源训练步数：`22000`

`policy.onnx` 保留完整源 actor，并增加四个语义输入。
训练 `.pt` 仅用于来源核验，不会复制到部署目录。
