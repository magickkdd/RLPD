# P4 规格书：Adroit（G4）

> 版本: 2026-10-17 v2（v22 扫尾）· 上游: REPRODUCTION_PLAN.md v22 · 预算: **1 冒烟 ≈ 0.2 run·h**（结果级 = 弹性，不入验收门）

## 1. 判据（G4）
1. `train_finetuning.py --env_name=pen-binary-v0` 跑通 1k 步（**mj_envs/mjrl/AWAC 依赖链全通**）
2. 结果级（**v21 降为弹性项，不入验收门**）：仅当 AWAC 原数据（路径①②）拿到才跑——路径③（hdf5 自转）存在稠密/二值奖励混批的尺度不一致问题（BinaryDataset 直接复制 npy 的 rewards，与 binary env 的二值在线奖励同批混合会使 critic 目标尺度不一致），且 pen-cloned 数百万条"预灌秒级"不成立

## 2. 事实基础（源码级核实）
| # | 事实 | 出处 |
|---|---|---|
| F1 | **`pen-binary-v0` env 由 mj_envs 提供**——d4rl 1.1 wheel 有 hand_manipulation_suite/（pen-human 等稠密版）但**无 -binary- 变体**、rlpd 无注册；README 明确装 mj_envs（philipjball fork）。**submodule 必须 pin 到 README 对应 commit，勿 --remote 拉最新**（依赖可复现性） | d4rl wheel 目录清单 + rlpd README L44-81 |
| F2 | mjrl (aravindr93) ✅ P0a 已装（--no-deps） | RESUME.md |
| F3 | **BinaryDataset（rlpd）读 AWAC npy**：`{env_prefix}2_sparse.npy`（pen → pen2_sparse.npy）+ `{env_prefix}_bc_sparse4.npy`；各取**前 90%** 为训练集（dataset_split=0.9）；`include_bc_data` 默认 True（FLAG binary_include_bc）；`remove_terminals=True` → masks 全 1；rlkit dict 格式（x["observations"] 为 list of dict） | binary_datasets.py L10-12、L72-90、L95-135 |
| F4 | **AWAC .npy 不在 imone/D4RL**（HF API 全清单核实）——数据源三路方案见 §4 | HF API 实测 |
| F5 | Adroit 超参（README + Table 2）：UTD=20、`backup_entropy=False`、3 层 `(256,256,256)`、CDQ 开（num_min_qs=2 默认）、1M 步（我们降 200k） | rlpd README L73-81 |
| F6 | BinaryDataset 的 env_prefix = `env.spec.name.split("-")[0]` = "pen" | binary_datasets.py L70 |

## 3. 依赖链（三件，按序，**比算法更难**）
| # | 依赖 | 动作 | 验收 |
|---|---|---|---|
| D4-a | **mj_envs**（未装） | 实例: `git clone --recursive https://github.com/philipjball/mj_envs.git && cd mj_envs && git submodule update --remote && pip install -e .`；**submodule 递归 clone 在实例网络易断 → 本机 clone（含 submodule 展开成普通目录）打 tarball scp 兜底** | `python -c "import mj_envs, gym; gym.make('pen-binary-v0')"` |
| D4-b | mjrl | ✅ 已装 | import mjrl |
| D4-c | **AWAC npy 数据**（未下） | 三路方案（§4） | `np.load('pen2_sparse.npy', allow_pickle=True)` + 键结构匹配 |

## 4. AWAC 数据三路方案（按序尝试）
| 路 | 方案 | 说明 |
|---|---|---|
| ① | README Google Drive zip（file id `1SvV...9Xx`）**本机下载 → scp** | 本机出境受限（Tailscale 协调服务器先例）→ 可能不可行；可试 gdown/代理 |
| ② | HF 换关键词再搜（awac-data / rlkit / adroit npy） | 已搜 "awac" 无效；再试组合关键词 |
（路径③细节见上表——v22 保持降级表述）：pen-cloned-v1.hdf5 等为**稠密奖励**，与 binary env 二值在线奖励混批尺度不一致（v21 降级：非"明确可行"）——若走此路必须逐步重标二值奖励（需 env 成功判据）或 `--binary_include_bc=False` 并如实声明偏离；因此结果级仅在路径①②成功时执行 | — |

## 5. 任务矩阵与命令模板（1 冒烟 + 3 结果级）
```bash
# G4 冒烟（v21：--start_training=0，1k 步必须覆盖 update 路径，防假阳性）：
cd /mnt/workspace/rlpd && export LD_LIBRARY_PATH=... && export WANDB_MODE=offline && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
/mnt/data/miniforge3/envs/rlpd311/bin/python train_finetuning.py \
  --env_name=pen-binary-v0 --utd_ratio=20 --start_training=0 --max_steps=1000 \
  --config=configs/rlpd_config.py --config.backup_entropy=False \
  --config.hidden_dims="(256, 256, 256)" --seed=42 --checkpoint_model=True \
  --project_name=rlpd_p4 --log_dir=/mnt/workspace/ckpt/p4/RLPD/pen-binary-v0/s42 \
  > /mnt/workspace/logs/p4_smoke_s42.log 2>&1
# 结果级（弹性，仅路径①②成功时）：同命令 --max_steps=200000 × seeds 42/43/44（~1.9h/run）
# 注: Adroit 走 BinaryDataset 对称采样（无预灌环节；预灌只存在于 train_baselines.py 的 sacfd 路径）
```
- 评估口径（v21 修正）：**pen-binary-v0 是 mj_envs 的 env，不是 d4rl OfflineEnv——没有 get_normalized_score 方法（调用会 AttributeError）**；pen horizon=100 且奖励二值 → 原始 return 恰好=成功时步占比（数值上等价百分数），报告用原始 return 并注明此等价性；door/relocate（horizon 200/1000）若开弹性项则不可用该等价

## 6. P4-W1 验证清单
| # | 验证 | 判据 | 失败处置 |
|---|---|---|---|
| V4-1 | mj_envs 装机 + env 注册 | gym.make('pen-binary-v0') 成功 | 走 tarball scp 路；numpy2 报错按栈小修（np.float 类） |
| V4-2 | 数据加载 | BinaryDataset(env) 构建 len>0；仿 smoke item 6 跑 agent.update 1 次 | 键结构不匹配 → 转换器修 |
| V4-3 | 1k 步冒烟（**--start_training=0**，v21 修正 start_training 陷阱——否则 1k 步内不触发任何 update，G4 假阳性） | 无报错 + evaluation/return 有数 + training/critic_loss 有数（.wandb 可查） | 按栈排查 |

## 7. 风险
| 风险 | 概率 | 对策 |
|---|---|---|
| mj_envs submodule 断链 | 高 | 本机打包 scp（D4-a 兜底明确） |
| AWAC 数据拿不到 | 中 | 路径①②尽力；③ 受限于奖励语义（稠密/二值混批）已降级——**结果级为弹性项，拿不到不影响 G4（冒烟门）** |
| mj_envs 老代码 numpy2 不兼容 | 中 | 按栈小修；mj_envs 是薄 wrapper 层 |
| pen 稀疏奖励 200k 不学习 | 中 | G4 只要求跑通；结果级为弹性；**预期校准：论文 Fig.10 Pen Sparse 面板 RLPD 200k 处约 20~30 分**（400~600k 才进 40~50），勿把 200k 偏低误判为实现问题 |

## 8. 批次衔接
依赖链准备（本机打包/数据搜索）可与 P2 W2 并行做；冒烟一旦通过，结果级 runs 插入 fleet 队列。
