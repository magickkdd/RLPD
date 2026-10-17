# P5 规格书：Pixels（G5）

> 版本: 2026-10-17 v2（v22 扫尾）· 上游: REPRODUCTION_PLAN.md v22 / P2_SPEC.md v2 · 预算: **校准 1 + 结果级 6 ≈ 15~18 run·h**（含 Online 组）
> 定位: **全方案 B 最高适配风险段**（numpy2 + dmcgym + dm_control 组合从未运行）——校准前置，失败可整体标"未复现"且不阻塞 P6/P7

## 1. 判据（G5）
- cheetah-run 跑通 1 个完整 run + RLPD > Online（同库对照）趋势成立
- **humanoid 默认不跑**（内存峰值 ~18GB 在线 buffer + 渲染负载；额度富余时降级 1M 步为弹性项）

## 2. 事实基础（源码级核实）
| # | 事实 | 出处 |
|---|---|---|
| F1 | **pixels 训练脚本与 locomotion 结构不同**：双 prefetch 队列（`ds_iterator` + `replay_buffer_iterator`，get_iterator 模式）→ 每步 `combine(offline_batch, online_batch)`；`--offline_ratio=0` 同样使 combine 崩溃 → **Online 基线必须派生 train_baselines_pixels.py**（不能复用 P2 的 train_baselines.py） | train_finetuning_pixels.py L120-148、L175-186 |
| F2 | `MemoryEfficientReplayBuffer`（像素 obs 大）与普通 ReplayBuffer 分支并存（FLAG memory_efficient_replay_buffer） | 同上 L131-148 |
| F3 | 内层循环 `range(1, max_steps // action_repeat + 1)`；**action_repeat：PLANET_ACTION_REPEAT[cheetah-run-v0] = 4（脚本默认，train_finetuning_pixels.py L58-65），论文 Table 1 规定 2——v21 决策：命令显式传 `--action_repeat=2` 对齐论文**（300k env-steps = 150k 内循环；官方默认 4 的差异写入 NB2） | 脚本 L58-65 + 论文 Table 1 |
| F4 | rlpd_pixels_config = drq_config 基 + `num_qs=10, num_min_qs=1, critic_layer_norm=True, backup_entropy=False`；CNN `(32,64,128,256)`/stride2/VALID/d4pg encoder/latent 50——CDQ 关、LN 开（论文 Table 2 Pixels 行 ✓）；**UTD 默认 1**（Fig.6 的 UTD=10 为弹性） | configs/rlpd_pixels_config.py、pixel_config.py、README L93-97 |
| F5 | V-D4RL 数据 = **conglu/vd4rl（HF 官方）**；落盘目录 **`~/.vd4rl/main/cheetah_run/medium/64px/*.npz`（env 名下划线！vd4rl_datasets.py L18 `"_".join(env_name.split("-")[:-1])`——连字符会读不到，v21 修正）** | vd4rl_datasets.py L16-20 |
| F6 | 渲染：无头实例必须 `MUJOCO_GL=osmesa`（apt 已装 libosmesa6-dev）；dmcgym 把 dm_control env 包成 gym | 计划 §8.2 + dmcgym |
| F7 | **numpy2 + dmcgym + dm_control 组合从未运行**（G0a 后）——最高风险项 | RESUME.md |

## 3. 任务矩阵（校准 1 + 结果级 6）
| 步骤 | 内容 | 产出 |
|---|---|---|
| P5-W1 校准 run | cheetah-run medium × **10k env-steps、--start_training=0（v21 修正：原 5k 步 + start_training=5000 会全程不触发 update——start_training 陷阱第三次复发）**（RLPD 配置） | 实测**含 CNN 前反向与 buffer 增长**的吞吐与内存峰值 → 锁定 300k 时长与并发 |
| 结果级 | {RLPD（原版脚本）, Online（D5-1 基线）} × 3 seeds × 300k | G5 判定 |

```bash
# RLPD 模板：
cd /mnt/workspace/rlpd && export LD_LIBRARY_PATH=... && export WANDB_MODE=offline && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && export MUJOCO_GL=osmesa && \
/mnt/data/miniforge3/envs/rlpd311/bin/python train_finetuning_pixels.py \
  --env_name=cheetah-run-v0 --start_training=5000 --max_steps=300000 \
  --config=configs/rlpd_pixels_config.py --seed=42 --checkpoint_model=True \
  --project_name=rlpd_p5 --log_dir=/mnt/workspace/ckpt/p5/RLPD/cheetah-run/s42 \
  > /mnt/workspace/logs/p5_RLPD_cheetah-run_s42.log 2>&1
```

## 4. 交付物 D5-1：`rlpd/train_baselines_pixels.py`（~40 行）
派生自 train_finetuning_pixels.py，改动点（F1）：
1. 删 `ds_iterator` 与 `combine(offline_batch, online_batch)`
2. **在线迭代器 batch = batch_size×utd×(1−offline_ratio) = 128 ≠ RLPD 组的 256——必须传 `--offline_ratio=0`（或改迭代器 sample_args）使 Online 组 batch=256，保持"唯一变量=有无离线数据"（v21 修正：否则 G5 结论不可用）**
3. `--baseline=online`（唯一模式；RLPD pixels 本尊 = 原脚本）；MemoryEfficientReplayBuffer 分支保留
**验收**: cheetah-run 1k env-steps 冒烟无 ds 加载痕迹、critic_loss 有限

## 5. 数据
| 位置 | 源 | 校验 |
|---|---|---|
| `~/.vd4rl/main/**cheetah_run**/medium/64px/*.npz`（下划线，F5） | conglu/vd4rl hf-mirror 直下（**只下 cheetah-run medium**，每任务几 GB） | npz 可读 + episode 数>0；分片加载内存峰值观察 |

## 6. P5-W1 验证清单
| # | 验证 | 判据 | 失败处置 |
|---|---|---|---|
| V5-a | **import 冒烟**（numpy2 + dmcgym + dm_control + osmesa） | import 链无错；env.reset() 出 64px 帧（shape 校验） | 按栈修；修不动 → P5 标"未复现"，不阻塞 P6/P7 |
| V5-b | 校准 run（**10k env-steps、--start_training=0**） | 含 CNN 前反向的吞吐/内存写入 RESUME；并发与时长锁定 | 吞吐 <5 steps/s → 300k 改 150k 并 NB 注明 |
| V5-c | 数据加载 | VD4RLDataset 读 npz 正常 | 按栈排查 |
| V5-d | D5-1 冒烟（online 1k） | critic_loss 有限 | — |

## 7. 风险
| 风险 | 概率 | 对策 |
|---|---|---|
| dm_control/dmcgym numpy2 不兼容 | **高（首跑）** | V5-a 前置；dm_control 侧风险大——按栈修或标"未复现"；**不阻塞其他阶段** |
| osmesa 渲染吞吐低 | 中 | **env-steps 固定 300k 时渲染总量与 action_repeat 无关**；ar=2（vs 官方 4）使梯度更新 75k→150k 更贵——V5-b 实测定夺 |
| 数据分片下载/内存 | 中 | 只下 cheetah-run；逐 episode 流式加载确认 |
| **像素 ckpt 磁盘占用**（pixels 侧 save_checkpoint 无 keep，每 run ~30 份 × CNN ensemble 2-5MB ≈ 100-150MB/run，7 runs ≈ ~1GB） | 低 | status.sh 的 du 输出监控；持久盘 1PB 空间充足；超出预期则在 pixels 脚本 save 处加 keep=20（一处小改） |
| Online 组挤压额度 | 中 | Online ×3 seeds 视额度弹性（砍到 2 seeds 需 NB 注明） |

## 8. 批次衔接
校准 run 可在 P2 W2 期间随时插入；结果级在 P5-W1 全过 + P2 终评后排队。
