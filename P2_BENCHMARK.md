# P2_BENCHMARK（数据 sanity + 论文参考线）

| 任务 | 数据文件 | transitions | best episode return | best norm (v0 REF) | 备注 |
|---|---|---|---|---|---|
| halfcheetah-expert-v0 | halfcheetah_expert.hdf5 | 1000000 | 11252.0 | 92.9 | expert 数据 best_norm<95（REF 即数据集最强 episode，正常应 97~100）——需关注数据强度 |
| hopper-expert-v0 | hopper_expert.hdf5 | 1000000 | 3759.1 | 116.1 | OK |
| walker2d-expert-v0 | walker2d_expert.hdf5 | 1000000 | 5011.7 | 109.1 | OK |
| walker2d-medium-replay-v0 | walker_mixed.hdf5 | 302000 | 4132.0 | 90.0 | OK |
| walker2d-medium-v0 | walker2d_medium.hdf5 | 1000000 | 4226.9 | 92.0 | OK |
| antmaze-umaze-v2 | Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5 | 1000000 | 1.0 | 100.0 | OK |
| antmaze-large-diverse-v2 | Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5 | 1000000 | 1.0 | 100.0 | OK |

## Fig.19 读图参考线（人工读图，±5 误差，仅参考不作判定；已写入 JSON 供 aggregate 引用）
- hopper-expert-v0 RLPD 终值 ≈ 100；walker2d-expert-v0 ≈ 105；halfcheetah-expert-v0 ≈ 100
- REDQ(Online) 在三任务上均有竞争力（论文 A.3：locomotion 纯在线可较快解决）
