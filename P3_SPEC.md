# P3 规格书：AntMaze + CDQ 消融（G3）

> 版本: 2026-10-17 v2（v22 扫尾）· 上游: REPRODUCTION_PLAN.md v22 / P2_SPEC.md v2 · 预算: **9 runs ≈ 30 run·h**
> **⚠️ v24 范围裁剪（2026-10-18 02:30，用户决策，额度约束触发）**
> 任务矩阵收缩为 **umaze ×3 seeds（3 runs ≈ 4.6h）**：large-diverse 的 CDQ 消融 6 runs 砍掉。
   **G3-2（CDQ 开 vs 关方向）标注"未复现（范围裁剪）"**；G3-1（umaze 成功率从 0 上升）照常判定。
   gen_runs --suite p3 输出 9 行中只跑 RLPD/umaze 3 行；终评同理。
> 判据影响：不依赖对照的主判据照常执行；依赖对照/消融的判据标注 **"未复现（范围裁剪）"** 入 NB2。
> 已完成资产保留：3 个 SACfD halfcheetah seeds（完整跑完，可作附录参考线）；续训体系见 RESUME.md §5.5。


## 1. 判据（G3，不得放宽）
1. ≥1 个 AntMaze 任务成功率从 ~0 上升（学习发生）——umaze 主结果
2. CDQ=开 vs 关 方向与 Fig.8 一致（**开更差**）——**v21 域修正：Fig.8 的 AntMaze 面板是 Large Diverse（论文 L284），umaze 上 CDQ 大概率无差异 → 消融域改 antmaze-large-diverse-v2**

## 2. 事实基础（源码级核实）
| # | 事实 | 出处 |
|---|---|---|
| F1 | **antmaze-umaze-v2**：`max_episode_steps=700`；**large-diverse-v2 = 1000**（时长按 1000 口径）、`ref_min=0.0 / ref_max=1.0`（get_normalized_score 输出即成功率百分数）、`v2_resets=True`、sparse reward | d4rl 1.1 wheel locomotion/__init__.py L325-336、infos.py L155/L245 |
| F2 | 数据本地文件名 = 数据集 URL basename：**`Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5`**；hf-mirror imone/D4RL 中同名（扁平根目录）——**下载即用无需改名** | d4rl offline_env.py L31-34 + HF API 清单实测 |
| F3 | reward−1 已内置：train_finetuning.py 两处 `"antmaze" in FLAGS.env_name → rewards -= 1`（pretrain L135-136、在线 L192-193） | rlpd 源码 |
| F4 | AntMaze 超参（论文 Table 2 + README）：CDQ 关（num_min_qs=1）、熵备份关（backup_entropy=False）、3 层 `(256,256,256)`、300k 步、UTD=20；**LayerNorm 保持开**（README 未覆盖） | rlpd README L31-42 + 论文 Table 2 |
| F5 | **numpy2 栈下 antmaze import 链（d4rl.locomotion.maps/maze_env）从未运行**——P3 最大未知项 | RESUME.md 已知坑 |

## 3. 任务矩阵与命令模板（9 runs ≈ 30 run·h）
| 组 | env | config 覆盖 | seeds | runs | 单 run |
|---|---|---|---|---|---|
| RLPD（CDQ 关 = AntMaze 主口径） | antmaze-umaze-v2 | rlpd_config + `--config.backup_entropy=False --config.hidden_dims="(256, 256, 256)" --config.num_min_qs=1` | 42/43/44 | 3 | ~3.3h |
| RLPD（CDQ 关，消融基线） | **antmaze-large-diverse-v2** | 同上 | 42/43/44 | 3 | ~3.3h |
| CDQ 开（消融，large-diverse 域） | antmaze-large-diverse-v2 | 同上 + `--config.num_min_qs=2` | 42/43/44 | 3 | ~3.5h（CDQ-on 略慢） |

```bash
# gen_runs.py --suite p3 模板（RLPD s42；CDQ-on 组仅 num_min_qs 与 log_dir/algo 不同）
cd /mnt/workspace/rlpd && export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/mnt/workspace/mujoco210/bin:$LD_LIBRARY_PATH \
  && export WANDB_MODE=offline && export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
/mnt/data/miniforge3/envs/rlpd311/bin/python train_finetuning.py \
  --env_name=antmaze-umaze-v2 --utd_ratio=20 --start_training=5000 --max_steps=300000 \
  --config=configs/rlpd_config.py --config.backup_entropy=False \
  --config.hidden_dims="(256, 256, 256)" --config.num_min_qs=1 \
  --seed=42 --checkpoint_model=True --project_name=rlpd_p3 \
  --log_dir=/mnt/workspace/ckpt/p3/RLPD/antmaze-umaze-v2/s42 \
  > /mnt/workspace/logs/p3_RLPD_antmaze-umaze-v2_s42.log 2>&1
```
- umaze 承载判据 1（成功率从 0 上升）；**CDQ 消融在 large-diverse 域**（Fig.8 原域，umaze 无信号风险高——v21 域修正）
- large-diverse 上 CDQ 关/开各 3 seeds，唯一变量 = num_min_qs 1→2（核查已确认干净单变量，无索引语义陷阱）

## 4. 数据
| hf-mirror 源文件 | 本地文件名 | 校验 |
|---|---|---|
| Ant_maze_u-maze_..._sparse_fixed.hdf5（umaze，~1GB） | 同名 | h5 直读 + reward∈{0,1} 验证 |
| **Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5**（large-diverse，~1-2GB） | 同名 | 同上 |

## 5. P3-W1 验证清单（先证假设再跑批）
| # | 验证 | 方法 | 判据 | 失败处置 |
|---|---|---|---|---|
| V3-1 | antmaze import 链在 numpy2 栈（F5） | `python -c "import gym,d4rl; gym.make('antmaze-umaze-v2')"` + env.reset() | 无错 | 按栈小修（多为 np.float 弃用类）；修不动 → P3 标"未复现"不阻塞他阶段 |
| V3-2 | 1k 步冒烟（--start_training=0） | 标准命令 max_steps=1000 | 无报错、reward∈{0,1}、evaluation/return 有数 | 按栈排查 |
| V3-3 | 成功率评估语义 | `env.get_normalized_score(0.5) == 0.5`（**d4rl 返回 0~1**，v21 修正）；evaluate() 对 antmaze 的 return = 每集回报（成功 1/失败 0）→ mean 即成功率（0~1），aggregate ×100 对齐 | 直通 | — |
| V3-4 | 数据直读 + 终止语义 | h5path 直读；抽查 done/mask | transitions>0；TimeLimit 与 success 的 mask 区分正确 | — |

## 6. 交付物
1. **gen_runs.py --suite p3**（9 行清单：umaze 3 + large-diverse CDQ off/on 各 3；v4 已实现并用 --config.num_min_qs）
2. **aggregate.py 扩展**：antmaze 判定源 = final_eval.json（**100 trials 为论文 B.1 对 AntMaze 的明确规定，非扩展口径**）；成功率口径自动识别
3. download_data.sh 已含 umaze + large-diverse 两个文件 ✓（v22 补）

## 7. 风险
| 风险 | 概率 | 对策 |
|---|---|---|
| numpy2 下 d4rl.locomotion 挂 | 中 | V3-1 前置；错误按栈修；修不动 → 如实报告"未复现" |
| umaze 稀疏奖励 300k 步不学习（成功率仍 0） | 中 | ① 核对 evaluate 成功率路径 ② 查 reward−1 与 v2_resets ③ 论文 umaze ~100k 内应上升——3 seeds 均 0 且 checklist 通过才允许下"方向不成立"结论 |
| **large-diverse 3 seeds 判"CDQ 开更差"可能反向**（最难任务、论文 10-seed 方差带宽；**max_episode_steps=1000**，时长按 1000 口径） | 中 | 反向时过 P2_SPEC §6.4 checklist；论文 Fig.8 的 CDQ 有害也在 large 域呈现，如实报告 |
| 数据 ~1-2GB ×2 下载慢 | 低 | hf-mirror 直下（文件名一致无需改名）+ scp 兜底；download_data.sh 已含两个文件 |

## 8. 批次衔接
P2 W2 期间 fleet 空位可直接追加 P3 runs（数据先下好）；P3-W1 冒烟不依赖 P2。
