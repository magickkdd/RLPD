# P6 规格书：消融矩阵（G6）

> 版本: 2026-10-17 v2（v22 扫尾）· 上游: REPRODUCTION_PLAN.md v22 / P2_SPEC.md v2 · 预算: **9 个新 runs ≈ 18 run·h**
> **⚠️ v24 范围裁剪（2026-10-18 02:30，用户决策，额度约束触发）**
> **整阶段砍掉（范围裁剪）**，NoLN / Seeded / Symmetric 消融全部不跑。
   **G6 两判据标注"未复现（范围裁剪）"**；claim 2（Seeded vs Symmetric）与 claim 3 的 LN 消融面在报告中如实标注。
> 判据影响：不依赖对照的主判据照常执行；依赖对照/消融的判据标注 **"未复现（范围裁剪）"** 入 NB2。
> 已完成资产保留：3 个 SACfD halfcheetah seeds（完整跑完，可作附录参考线）；续训体系见 RESUME.md §5.5。


## 1. 判据（G6，不得放宽）
1. LN off vs on：**expert locomotion 上 no-LN 曲线劣于 LN（样本效率与终值）**——复现对象 = **Fig.19**（A.3 expert 数据消融，"LayerNorm can marginally help both sample efficiency and asymptotic performance"，论文 L558；legend 含 no LN）；**claim 3 的 Fig.2/Fig.7 域（AntMaze Large Play / Pen Sparse / Adroit / Humanoid）不在矩阵，如实标注"未覆盖"（v21 修正）**
2. Seeded vs Symmetric：**论文 Fig.11 的原主张 = 终值相当 + Symmetric 样本效率更优 + 方差更低**——判据改口径为「到达 60 分所需步数（样本效率）+ seed 间 std」**而非终值比较**（v21 修正：终值判定与论文主张不符，200k 终点本应接近）
- offline_ratio 消融（Fig.12，claim 4）= 弹性项，不进 G6

## 2. 事实基础
| # | 事实 | 出处 |
|---|---|---|
| F1 | **Seeded(RLPD-config) ≠ SACfD**：Seeded = RLPD 全配置（E=10/LN/CDQ 开）只换 buffer 策略（Fig.10 的 "Seeded"）；SACfD = plain SAC + seeded（P2 已跑，Fig.4 口径）——两者必须分开命名与对账 | P2_SPEC F2 + 论文 §5 L233 |
| F2 | Fig.11 的任务是 **walker2d-medium-v0**（非 medium-replay、非 expert） | RLPD_paper.md L315 |
| F3 | LN 消融 = `--config.critic_layer_norm=False` 纯 flag 覆盖，**零新代码** | rlpd config_flags |
| F4 | Seeded 复用 `train_baselines.py --baseline=sacfd` 通路但**显式传 `--config=configs/rlpd_config.py`**（覆盖默认 sac_config）；固定容量 = ds_size 循环覆盖 | P2_SPEC D1 |
| F5 | LN-on 组（walker2d-expert ×3 seeds）**复用 P2 RLPD runs**，不重跑 | 复用设计 |
| F6 | LN 的主要收益之一是**方差抑制**（论文 Fig.2 讲 Q 值发散）——对账表必须显式呈现 std 列 + Q 值曲线对比 | 论文 Fig.2/L73 |

## 3. 任务矩阵（9 个新 runs）
| 消融 | 任务/步数 | config | 脚本 | 新 runs | 复用 |
|---|---|---|---|---|---|
| LN off | walker2d-expert-v0 / 200k | rlpd_config + `--config.critic_layer_norm=False` | train_finetuning.py | 3（s42-44） | LN on ×3 = P2 RLPD runs |
| Seeded | walker2d-medium-v0 / 200k | rlpd_config（LN on 默认） | train_baselines.py --baseline=sacfd | 3 | 无 |
| Symmetric | walker2d-medium-v0 / 200k | rlpd_config | train_finetuning.py | 3 | 无 |

```bash
# LN-off 模板（相对 P2 RLPD 模板仅加一个 config 覆盖 + log_dir/algo 改名）：
... train_finetuning.py --env_name=walker2d-expert-v0 --utd_ratio=20 --start_training=5000 \
  --max_steps=200000 --config=configs/rlpd_config.py --config.critic_layer_norm=False \
  --seed=42 --checkpoint_model=True --project_name=rlpd_p6 \
  --log_dir=/mnt/workspace/ckpt/p6/NoLN/walker2d-expert-v0/s42 \
  > /mnt/workspace/logs/p6_NoLN_walker2d-expert-v0_s42.log 2>&1
# Seeded 模板：
... train_baselines.py --baseline=sacfd --config=configs/rlpd_config.py \
  --env_name=walker2d-medium-v0 --utd_ratio=20 --start_training=5000 --max_steps=200000 \
  --seed=42 --checkpoint_model=True --project_name=rlpd_p6 \
  --log_dir=/mnt/workspace/ckpt/p6/Seeded/walker2d-medium-v0/s42 \
  > /mnt/workspace/logs/p6_Seeded_walker2d-medium-v0_s42.log 2>&1
# Symmetric = train_finetuning.py 原版（同 env/步数）
```

## 4. 数据
| 任务 | hf-mirror 源 | 本地名 | 状态 |
|---|---|---|---|
| walker2d-expert-v0 | walker2d_expert-v2.hdf5 | walker2d_expert.hdf5 | P2 已下 ✓ |
| walker2d-medium-v0 | walker2d_medium-v2.hdf5 | walker2d_medium.hdf5 | download_data.sh 已含 ✓ |

## 5. P6-W1 验证清单
| # | 验证 | 判据 | 失败处置 |
|---|---|---|---|
| V6-1 | Seeded 预灌正常（walker2d-medium 数据） | [seeded] 日志条数 = 数据集条数、buffer_len 正确 | 按栈排查 |
| V6-2 | LN off 1k 步（**--start_training=0 防空转**，v21） | critic 数值有限（no LN 数值可能大——记录量级） | 按栈排查 |
| V6-3 | walker2d_medium.hdf5 直读 | transitions>0 | 重下 |

## 6. 交付物
1. **gen_runs.py 扩展 `--suite p6`**（9 行清单）
2. **aggregate.py 扩展**：消融对账表（LN on vs off：mean±std 双列 + Q 值曲线；Seeded vs Symmetric：同格式）——判定数据源同 P2（final_eval.json 100 eps + 训练曲线形状）

## 7. 风险
| 风险 | 概率 | 对策 |
|---|---|---|
| no LN 在 walker2d-expert 数值发散 | 中 | 如实记录（崩溃点步数 + 最后有效值），不算 run 失败；**但注意 walker2d-expert 的 NaN 不能作为 Fig.2（Q 发散，域不同）的复现证据——claim 3 的 Fig.2/7 部分标"未覆盖"** |
| Seeded vs Symmetric 终值接近 | **高——论文主张就是终值相当** | 判据用样本效率（到 60 分步数）+ std；**std 仅描述性呈现（n=3 相对误差 ~40%，不作显著性检验）**；aggregate v4 按此实现 |
| 复用判定错误（拿错 P2 runs） | 低 | gen_runs --suite p6 输出复用清单（log_dir 列表）供人工核对 |

## 8. 批次衔接
LN-off 与 Seeded/Symmetric 依赖 P2 的 walker2d-expert runs 产出（终值参照）——排在 P2 W2 完成后接队；数据可提前下好。
