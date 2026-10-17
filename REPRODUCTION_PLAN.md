# RLPD 论文复现计划

> 论文: Efficient Online Reinforcement Learning with Offline Data (Ball, Smith, Kostrikov, Levine; ICML 2023)
> 代码: https://github.com/ikostrikov/rlpd (官方 JAX 实现)
> 本文档为讨论基线，随讨论持续迭代。变更记录见文末。

---

## 1. 目标与验收标准

完整复现论文核心结论（不只是跑通代码）：

| # | 论文主张 | 出处 | 验收标准 |
|---|---|---|---|
| 1 | RLPD 超越 Prior SoTA | Fig.4 / 附录A | 各套件聚合曲线达到附录 A 量级 |
| 2 | 对称采样优于 buffer 初始化 | Fig.10/11 | **Seeded(RLPD-config) vs Symmetric**（样本效率到 60 分步数 + std 描述性）+ RLPD vs SACfD 如实报告（SACfD=UTD1+plain SAC，论文定义，多变量对照） |
| 3 | LayerNorm 防止 Q 值发散 | Fig.2/7/19 | **Fig.19 谱系**（expert locomotion：no-LN 样本效率/终值劣化）；Fig.2/7 的域标注"未复现" |
| 4 | 采样比例不敏感，50% 为折中 | Fig.12 | 0/25/75/100% 消融趋势一致 |
| 5 | 环境特定设计工作流（CDQ 在 AntMaze 必须关） | Fig.8 / Table 2 | 反直觉结论可复现 |
| 6 | Pixels 域可迁移 + 高 UTD 增益 | Fig.5/6 | RLPD > Online（同库对照） |

**Claim × 阶段覆盖矩阵（v21 回写，与规格对齐——防止"验收合同"与"实际交付"两套口径）**：

| claim | 论文图 | 覆盖阶段 | 覆盖程度 | 缺口声明 |
|---|---|---|---|---|
| 1 | Fig.4（21 环境） | P2（4 locomotion 任务子集）+ P3（AntMaze）+ P4（pen 冒烟） | **子集** | 无聚合曲线；Fig.4 标"4 任务子集" |
| 2 | Fig.10/11 | P2（SACfD 对照）+ P6（Seeded vs Symmetric） | 谱系覆盖 | SACfD=UTD1+plain SAC（论文定义）；Seeded(RLPD-config) 补变量隔离 |
| 3 | Fig.2/7/19 | P2+P6（Fig.19 谱系：expert locomotion LN 消融） | **部分** | Fig.2（AntMaze/Pen 域）、Fig.7（Adroit 等 4 域）标"未复现" |
| 4 | Fig.12 | 弹性项 | 未承诺 | 标注"弹性" |
| 5 | Fig.8 | P3（large-diverse 域面板子集） | 子集 | 仅 umaze 主结果 + large-diverse 消融域 |
| 6 | Fig.5/6 | P5（cheetah-run medium 单任务；UTD=1） | **子集** | humanoid/walker-walk/BC 基线/Fig.6 UTD 增益标"未复现" |

## 2. 资源盘点

### 2.1 本机（仅开发机，不参与训练 ✅ D10）
- GPU: RTX 3050 Laptop 4GB（弃用于训练，全部实验上云）
- 定位: 写代码、看 wandb 曲线、整理报告与文档，不消耗训练预算

### 2.2 ModelScope AMD 实例（全项目主力）
- 规格: 实测 **23 核** / **200GB RAM** / **192GB 显存**（MI300X 级，gfx942）
- 镜像: `ubuntu22.04-rocm7.2.3-py312-torch2.12.0`；训练环境为自建 conda py3.11（§8.2）
- 剩余额度: **100 GPU 小时**（可通过激励计划扩至 ~400h，见 2.4）；**实际余额以 W1 开机控制台读数为准（P2-SPEC V1），不推算**
- 200GB 内存 → humanoid 像素缓冲无压力（但 Pixels 路径从未在此 numpy2 组合下运行过，P5 需先冒烟）
- 适配已完成（v13-16 实测）：**jax 0.9.2 ROCm PyPI 四件套**（非 0.8.x 路线）、mujoco-py 按 numpy2 重编、d4rl 1.1 + mjrl + dmcgym

### 2.3 ModelScope NVIDIA A10 实例
- **不使用（D10）**：全部实验只在 AMD 实例执行，A10 的 36h 额度留作应急储备，不主动消耗

### 2.4 AMD x 魔搭开发者激励计划（时长扩充路径）
> 规则: https://modelscope.cn/events/247/开发者激励计划玩法说明 （报名截止 2026-12-31）

| 路径 | 要求 | 奖励 | 上限 |
|---|---|---|---|
| 前置注册 | AMD 开发者计划注册（魔搭账号授权登录） | +100h | +100h |
| 路径一: 技术文章 | 「开发者实践」板块原创文，标签"AMD GPU激励计划"，审核 ~5 工作日 | +25h/篇 | +50h |
| 路径二: Notebook | 「Gallery」板块 Notebook 实践（全部 cell 可跑通 + 注释完整 + 环境说明），标签同上 | +50h/篇 | +150h |

**与我们项目的映射（复现即内容，内容换算力）**：
- NB1（P0 产物）: 老版 JAX/flax/d4rl 栈在 ROCm 上的适配实践 → +50h
- NB2（P2 产物）: RLPD 复现之 Locomotion 主结果（训练曲线 + 评估） → +50h
- NB3（P6 产物）: RLPD 消融复现（LayerNorm / 对称采样） → +50h
- 文1（P0 后）: ROCm 环境踩坑记 → +25h
- 文2（P7 后）: 复现方法论与结果分析 → +25h
- 合计潜在 +275h，审核周期 ~5 工作日/篇 → **NB1/文1 在 P0 完成后第一时间提交**

### 2.5 个人云账号（可选扩展）
- 仅当激励计划额度 + 免费额度仍不足时启用（方案 B 下大概率不需要）

## 3. 技术路线（已确认）

- **官方 JAX 仓库直接复现**；消融优先命令行 flag；**例外（v17）：P2 对照基线需新增 train_baselines.py 改采样路径**（`offline_ratio=0/1` 在 combine() 必崩，Online 无法用 flag 实现；SACfD 需 seeded buffer + plain SAC config——不改算法逻辑，只改数据通路）
- wandb **offline 训练 + 定期 sync 校验**（实例实测 offline 可靠；聚合主源 = final_eval.json + 自研 .wandb 解析，云端导出双源对账防丢字段）
- 需自写的少量胶水代码：
  - **train_baselines.py**（≤50 行：--baseline=sacfd = plain SAC(configs/sac_config) + seeded 固定容量 buffer；--baseline=online = rlpd_config 无离线数据）+ final_eval.py（restore_checkpoint → 100 eps 终评）
  - ~~fleet 并发调度器~~ **xargs 模式**：`xargs -P 6 -d '\n' -I CMD bash -c CMD < runs.txt`（每行 `cd /mnt/workspace/rlpd &&` 开头）；**并发倍率待 W1-V4 重测**（2k 步表观 2.53× 被启动/编译/i=0-eval 污染，稳态口径 ≈1.33×，真值 1.33~4× 待 20k 步重测+rocm-smi 证据）
  - 结果聚合脚本 aggregate.py（解析 .wandb → 对账表格）+ gen_runs.py + status.sh
  - 数据下载：d4rl 走 hf-mirror（v2 数据改名喂 -v0 env，best-in-dataset sanity 校验）；V-D4RL 走 hf-mirror（Google Drive 方案作废）；Adroit npy 含 {env}_bc_sparse4.npy

## 4. 算力预算（P1 实测校准）

> 表头"估时"为单 run 独占实例的墙钟时间。**并发倍率 v17 修正：2k 步表观 2.53× 被启动/编译/i=0-eval 非稳态开销（~50%）污染，稳态口径 ≈1.33×（区间 1.3~1.6×）；但 GPU 算力远未打满（瓶颈疑为主机侧），真值待 W1-V4 重测（20k 步 + rocm-smi），在 1.35~1.6× 区间内取保守下限做预算（v23：删无测量支撑的 4×）**。额度按实例在线时长扣除。

| 类型 | 单 run 计算量 | 单 run 独占估时 (MI300X, P1 实测) |
|---|---|---|
| Locomotion RLPD (200k~250k × UTD20) | 4~5M 梯度步 | 1.9~2.4h（实测 33ms/env-step） |
| AntMaze RLPD (300k × UTD20) | 6M | ~3.5h（3 层网络略慢） |
| Adroit RLPD (1M × UTD20) | 20M | ~9.5h（**默认不跑全量**，见下方最小结果集） |
| SACfD (plain SAC + seeded, **UTD=1**, v21 按论文 §4.5 定义重定) | 0.2~0.25M 梯度步 | **~0.15h**（2ms/step；v17 的 UTD20 口径作废——§4.3 高 UTD 属论文 Green 核心设计，SACfD"无附加设计"即 UTD=1） |
| Pixels cheetah (300k, UTD1, CNN, **action_repeat=2 论文口径**) | 0.3M | 1~2.5 h（未实测；P5 校准 run 10k 步含 update 路径） |
| Pixels humanoid (3M, UTD1) | 3M | 6~10 h（未实测；**默认降级不跑**） |

### 验收门最小结果集（v17 重算——方案 B 的真实口径，非全量矩阵）

| 阶段 | 结果级 runs | run·h |
|---|---|---|
| P2 | 36（4 任务 × 3 算法 × 3 seeds；**SACfD=UTD1 plain SAC ≈0.15h/run**） | ~26 |
| P3 | 9（umaze RLPD 3 + **large-diverse CDQ off/on 各 3**——Fig.8 原域，v21 域修正） | ~30 |
| P4 | 1 冒烟（--start_training=0 防假阳性；**结果级降为弹性**——AWAC 数据不确定性） | ~0.2（弹性另计） |
| P5 | 3（cheetah-run medium 300k；humanoid 降级不跑） | ~7.5 |
| P6 | 9（LN-off × walker2d-expert ×3（on 复用 P2）；Seeded vs Symmetric × walker2d-medium ×6） | ~18 |
| **合计** | **62 runs** | **~113 run·h → 71~84 实例小时（÷1.6~÷1.35；v22 含 P2 Online 组重算，删 4× 无依据上界）** |

- 额度覆盖（v22）：71~84 实例小时（+W1/W3/余量）**逼近 100h 现余额度** → **P0b 注册 +100h 为必需项**；W1-V4 锁定倍率 + V1 读实际余额后复核；压缩顺序见风险提示
- 扩展 run（AntMaze 全 6 任务、Adroit 1M、humanoid、5 seeds）= 弹性项，仅在剩余额度富余时追加，不进验收门

> 注（P1 实测 + v17 修正）：单路 UTD20 ≈ 30 steps/s、UTD1 ≈ 500 steps/s，单梯度步 ~1.7-2ms。2k 步并发曲线 1/2/4/6 路 = 168/193/334/398s，表观吞吐 2.53×@6 路**含 ~50% 非稳态开销（import+编译+i=0 eval），稳态口径 ≈1.33×，真值待 W1-V4 重测**。"单 run 把 GPU 喂饱"系无 rocm-smi 证据的推测：30 steps/s × UTD20 的算力需求（每秒数 GFLOPs）远低于 MI300X 能力，瓶颈疑为主机侧（Python 循环/combine() 交织拷贝/H2D）。像素域 dm_control 渲染为 CPU 瓶颈，2~3 路为宜。

### 全量忠实矩阵（10 seeds）≈ 3000+ GPU 时
| 套件 | runs | GPU 时 |
|---|---|---|
| Locomotion (12×10, RLPD+SACfD) | 240 | ~630 |
| AntMaze (6×10, RLPD+SACfD) | 120 | ~390 |
| Adroit (3×10, RLPD+SACfD) | 60 | ~525 |
| Pixels (6 任务×10, RLPD+Online+BC；DrQ-v2 引用论文值) | 180 | ~520 |
| Fig.6 UTD=10 cheetah-expert | 20 | ~60 |
| 消融矩阵 (Fig.7/8/9/12) | ~150 | ~600 |

### 已选方案 B（简历级，全云端执行 ✅ D10）与额度覆盖
- 规模（v22 统一口径）: **验收门最小结果集 62 runs ≈ 113 run·h**（P2 50 + P3 30 + P4 0.2 + P5 15 + P6 18；上表）+ 弹性扩展项；历史数字（76/146、89/130 等）作废
- 额度来源（全部在 AMD 实例消耗）:
  | 来源 | 额度 | 说明 |
  |---|---|---|
  | AMD 实例现有 | 100h | **实际余额以 W1 开机控制台读数为准**（P0a/P1 在线时长 >6h 系推算，不可靠） |
  | AMD 注册奖励 | +100h | 完成开发者计划注册即得 |
  | 激励计划内容 | +275h（潜在） | 3 Notebook + 2 文章，随阶段产出陆续到账 |
- **风险提示（v22 重算）**: 113 run·h ÷ 1.35 ≈ 84 实例小时、÷1.6 ≈ 71——**加 W1 4-5h + W3 2h + 重跑余量后逼近/超出 100h 现余额度**（W1 开机读实际余额确认）。对策：① W1-V4 锁定倍率后立即复核；② **P0b 注册 +100h 为必需项**，NB1/文1 前置；③ 压缩顺序：P5 humanoid（已砍）→ P4 结果级（已弹性）→ Online 组 2 seeds → P6 Seeded seeds 3→2

## 5. 分阶段计划与验收门

| 阶段 | 内容 | 验收门（可判据） | 场地 |
|---|---|---|---|
| P0a | AMD: 按 §8.2 安装单建 py3.11 rocm 栈 + 冒烟（10 项）**✅ 已完成（2026-10-17）**：冒烟 9 PASS + 1 SKIP（8-modelscope 可选）；G0a 实测 `--max_steps=1000 --start_training=0` 全链路跑通，critic_loss 51.5→2.5、q 0.5→43、eval/return=196.7，健康 SAC 早期特征；注意默认 `start_training=10000`，1k 步空转属正常退出 | **G0a** ✅ 通过（实测记录见 RESUME.md） | AMD |
| P0b | 用户完成 AMD 开发者计划注册（+100h）；提交 NB1/文1（ROCm 适配实践） | 注册后额度 +100h 到账；NB1/文1 提交回执 | 魔搭 |
| P1 | 校准: 单 run 吞吐 + 并发曲线（1/2/4/6 路）**✅ 基本完成（2026-10-17，有保留）**：UTD20 单路 33ms/env-step 实测 ✓；但 2k 步并发曲线含 ~50% 非稳态开销，表观 2.53× 不可采信（稳态 ≈1.33×）——**W1-V4 用 20k 步重测 + rocm-smi 后 G1 才算完全关闭**；xargs 跑批模式已定 | **G1** ✅*（带 V4 复测尾巴） | AMD |
| P2 | Locomotion 结果 + SACfD/Online 对照 **（规格：[P2_SPEC.md](P2_SPEC.md) v2.1——4 任务 × 3 算法 × 3 seeds = 36 runs ≈ 50 run·h；SACfD = plain SAC + seeded + **UTD=1**（论文 §4.5 定义），Online = REDQ 无离线数据）** | **G2（v2 重设）**: ① halfcheetah-expert-v0 3-seed 终值归一化分（100 eps 终评）≥ 90（D4RL REF 绝对线）；② 3 expert 任务 RLPD 50k 步内达 60+ 分（Fig.19 形状）；③ RLPD vs SACfD / Online 对照如实报告（expert 任务 Online 有竞争力系论文自认，A.3）；④ 反向结果先过实现正确性 checklist | AMD |
| P3 | AntMaze + CDQ 消融（**规格：[P3_SPEC.md](P3_SPEC.md)**：antmaze-umaze-v0 ×6 runs；CDQ-off 复用主结果）**（numpy2 栈下 antmaze 路径未验证，先冒烟）** | **G3**: ≥1 个 AntMaze 任务成功率从 ~0 上升（学习发生）；CDQ=开 vs 关 方向与 Fig.8 一致（开更差） | AMD |
| P4 | Adroit（**规格：[P3_SPEC.md](P3_SPEC.md)**：mj_envs 依赖链 + pen2_sparse.npy/pen_bc_sparse4.npy 两类数据文件；G4 跑通 1k 步；结果级 = pen-binary 200k × 3 seeds，非 1M 全量） | **G4**: pen-binary-v0 跑通 1k 步（依赖链全通）；可选结果级出 pen 曲线 | AMD |
| P5 | Pixels（**规格：[P3_SPEC.md](P3_SPEC.md)**：数据走 conglu/vd4rl HF 官方镜像；humanoid 默认砍；**先 5k 步校准 run** 测 CNN/osmesa 吞吐与内存；Online 对照需检查 pixels 脚本 combine 路径） | **G5**: cheetah-run 跑通完整 run，RLPD > Online（同库对照）趋势成立 | AMD |
| P6 | 消融（最小集: LN on/off × walker2d-expert ×3；Seeded(RLPD-config) vs Symmetric × walker2d-medium ×3）**（Seeded≠SACfD，见 P2_SPEC F2）** | **G6**: LN 开关方向与 Fig.7/19 一致（no LN 性能降/方差大）；Seeded vs Symmetric 方向与 Fig.10/11 一致（对称采样优） | AMD |
| P7 | 汇总对账 + 复现报告（**规格：[P3_SPEC.md](P3_SPEC.md)**：11 张图表三态表 + NB2/文2 提交） | **G7**: 每张复现图表标注"一致/偏差/未复现"三态结论；偏差项有假设分析 | 本机整理 |

## 6. 实验配置速查（论文 Table 1/2 + README）

通用: batch 128+128（对称采样）, lr 3e-4, γ=0.99, E=10, ρ=0.005, UTD=20, LayerNorm=True

| 套件 | CDQ (num_min_qs) | 熵备份 | MLP | 其他 |
|---|---|---|---|---|
| Locomotion | 2 (开) | 开 | 2 层 | Fig.4 主结果 250k；Fig.19 expert 消融 200k；-v0 数据 |
| AntMaze | 1 (关) | 关 | 3 层 | 300k 步, reward−1, -v2 数据 |
| Adroit | 2 (开) | 关 | 3 层 | 1M 步, binary 数据 |
| Pixels (DMC) | 1 (关) | 关 | 2 层 | UTD=1 (Fig.6 用 10), action_repeat=2, 64px, 栈 3 帧 |

## 7. 数据清单

| 数据集 | 内容 | 来源 | 落盘位置 |
|---|---|---|---|
| D4RL Locomotion (-v0) | 按需: halfcheetah/hopper/walker2d × expert；walker2d medium-replay；后续任务按需追加 | **hf-mirror（imone/D4RL，v2 数据改名喂 -v0 env）**，Berkeley 源仅兜底；下载后 h5 直读 + best-in-dataset sanity | `/root/.d4rl/datasets/`（符号链接到 `/mnt/workspace/.d4rl` 持久盘） |
| D4RL AntMaze (-v2) | umaze / umaze-diverse / medium-play / medium-diverse / large-play / large-diverse | 同上 | 同上 |
| Adroit | pen/door/relocate 演示 .npy **+ {env}_bc_sparse4.npy**（binary_datasets.py 需要） | **Google Drive 原始 zip（本机下载 scp）；imone/D4RL 无 npy（已核实，无 hf-mirror 兜底）**；兜底 = pen hdf5 自转（受奖励语义限制，结果级弹性） | `~/.datasets/awac-data/`（持久盘） |
| V-D4RL 64px | cheetah-run medium（P5 最小集；其余弹性） | **conglu/vd4rl（hf-mirror）** | `~/.vd4rl/main/**cheetah_run**/medium/64px`（**下划线**，vd4rl_datasets.py L18） |

## 8. 环境与依赖

### 8.1 本机（**不维护** ✅ D10：本机不跑实验；空 env 保留无意义，以下仅存档）
- conda env `rlpd` (py3.9.23) 闲置；requirements.txt/本地 jax[cuda] 0.3.x 方案作废
- 本机只做：写代码、看 wandb、整理报告（无 GPU 依赖）

### 8.2 ModelScope AMD 实例（MI300X / ROCm 7.2.3）—— ✅ 实测打通（2026-10-17，v14 按第三方核查报告修正）

**实测结论（覆盖此前全部理论调研）**：
- 实例规格实况: 23 核 / 200GB RAM / 192GB 显存（MI300X, gfx942）/ 持久盘 NFS `/mnt/data`（与 `/mnt/workspace` 同一文件系统）/ 镜像 ROCm 7.2.3
- **`JAX_DEVICES [RocmDevice(id=0)]` 验证通过**；mujoco-py 2.1.2.14 在 py3.11 编译并导入成功
- 训练环境锁定 **py3.11**（唯一交集）: jax 0.9+ ROCm 栈要求 py≥3.11；mujoco-py 要求 py≤3.11（3.12 挂 `import distutils`，实测复现报告 A 系结论；3.10 无 ROCm wheel）
- **d4rl 版本选择（核查 A1 修正）**: git master 的 setup.py 有 `python_requires=">=3.7,<3.11"`（py3.11 pip 拒装，报告属实）；**改用 PyPI `d4rl==1.1`**——无 requires_python 上限、依赖 `gym<0.25 + mujoco-py + dm_control + h5py`（无 tensorflow、无 mjrl@git），与本项目 pin 全兼容
- **JAX ROCm 安装方式（核查 A2/A3 修正）**: `rocm-jax-v0.8.2` release 的 GitHub wheel URL 已 404（资产实名为 `+rocm7.2.0` 本地标签、cp311~cp314，计划中 cp310 写法两头皆错）；PyPI 上 jax-rocm7-plugin 无 0.8.x 版本。**正确方案 = PyPI 四件套同版本 pin**（无需任何 GitHub wheel）:
  ```bash
  pip install jax==0.9.2 jaxlib==0.9.2 jax-rocm7-plugin==0.9.2 jax-rocm7-pjrt==0.9.2
  pip install 'flax==0.9.0' optax ml_collections absl-py wandb tqdm imageio moviepy 'scipy<1.14' dm_control
  pip install gym==0.23.1 mujoco-py==2.1.2.14 'cython<3' 'setuptools<66' 'numpy<2' wheel   # 构建链先行
  pip install d4rl==1.1
  pip install 'git+https://github.com/ikostrikov/dmcgym'   # train_finetuning.py 顶层 import 必需（曾漏）
  ```
- **tensorflow 不装**（核查 B4）: d4rl 数据加载走 h5py，setup.py 无 tensorflow/tfds 依赖，此前计划有误
- 依据: `deployment/requirements-rocm.txt`（实测 pin）+ RESUME.md apt 清单与 mjrl/dmcgym 特殊装法；**旧脚本 install_step1.sh / setup_amd.sh / install_step2.sh(04:47 版) 已过期（py3.12/flax0.9/tensorflow-cpu/dmcgym master 等错误），W1 清理或加 DEPRECATED 头**

**适配改动清单（`rocm-adapt` 分支，代码文件 4 个 +67/-29 行，算法逻辑零改动）**：
1. ✅ `tanh_normal.py` / `tanh_transformed.py`: tfp → 纯 JAX 自实现。**对拍现状（核查 F4 诚实化）**: 6 个用例 = 5 passed + 1 xfailed；等价性覆盖 **unbatched log_prob（max diff ≈1e-6）/ mode / log_prob 网格 / 采样统计矩（含走 sample() API 的新用例）**；**不覆盖**: batched 同 seed 逐位一致（tfp key 派生不同，已恢复为 xfail 并注明）；**已知差异**: |a|→1 尾部 tfp 给 -inf/NaN、自实现 clip 在 1e-6（有限大负值，更稳）。参考环境 rlpd-ref = py3.9 + jax 0.3.25 + flax 0.6.2 + tfp 0.19 + scipy==1.10.1（jaxlib 0.3.x 已从 PyPI 下架，需 GCS 归档源）
2. ✅ `dataset.py`: `jax.tree_map` → `jax.tree_util.tree_map`（1 行）
3. ✅ `ensemble.py`: `FrozenDict.copy(add_or_replace=)` → 显式构造（防御性；drq_learner 同名 API 由冒烟项 3 实测）
4. ✅ `requirements-rocm.txt` 重写为实测 pin（含 dmcgym 补漏、删 tensorflow-cpu）

**运行时对策（核查 B2，不改上游代码）**:
- checkpoint 落点: absl 内置 `--log_dir` 默认空串 → `os.path.join('', prefix)` 退化为 cwd 相对路径。**对策**: 训练命令显式传 `--log_dir=/mnt/data/ckpt/<env_name>/s<seed>`（落持久盘）
- 并发命名冲突: exp_prefix 只含 seed 不含 env_name → fleet 调度器必须为每个 run 传独立 `--log_dir`（列为本 fleet 实现的验收项）

**冒烟测试（核查 B3 修复后 10 项）**: sys.path 缺陷已修（任意 cwd 可跑）；item 10 = `train_finetuning` 模块级导入；**item 8 modelscope = 可选件，SKIP 合规（v17 统一口径：9 PASS + 1 SKIP 即判通过）**。G0a = 冒烟通过 + `--log_dir` 指向持久盘跑 1k 步 ✅（已达成）

**静态排查与实战坑（含 v8 + 实战新增）**:
- `jax.tree_map`（dataset.py）1 行必改 ✅；`PRNGKey` 为 legacy-style key 但未弃用，jax 0.9.x 可用（核查 F6 措辞修正）
- mujoco-py apt 系统库: `libgl1-mesa-dev libglew-dev libosmesa6-dev libegl1 patchelf`（实测已装）
- dm_control 渲染: EGL on AMD 实例待验证 → `MUJOCO_GL=osmesa` 兜底
- `conda activate` 在 nohup 非 shell 不可靠 → 一律绝对路径 `$ENV/bin/python|$ENV/bin/pip`
- pip 守卫误判教训: 系统 py3.12 自带 jax/flax 干扰判断，安装前必须核对目标 env
- GitHub clone `curl 16 HTTP2 framing` → `git config --global http.version HTTP/1.1`
- 网页终端偶发冻结（前端 websocket 断连）→ 刷新重开；长命令一律 nohup + log 轮询

**评估口径（v17）**: 训练中每 5k 步 eval 10 episodes（代码默认）；终评 100 eps（`final_eval.py`）。**注：论文 B.1 的 100 trials 仅规定 AntMaze/Adroit，locomotion 用 100 eps 系本复现扩展口径，NB2 注明**

**断点续训**: 长训 run 开 `--checkpoint_model`（orbax，keep=20）+ wandb 实时上报；中断最坏代价 = 重跑单 run

**持久化策略（v12，实测确认）**: 环境在 `/mnt/data/miniforge3/envs/rlpd311`、mujoco210、数据集目录 `.d4rl/.vd4rl` 全在 NFS 持久盘；`$HOME` 仅符号链接；重启恢复见 `deployment/RESUME.md`；冒烟项 9 报告持久盘余量（数据集 ~20-40GB 对账待做）

**安全规则（核查"5 个先决问题"第 5 条整改）**: **任何上传到 ModelScope 的 bundle 永不包含 `.secrets.env`**（ms_upload.py 的安全缺口已修复并实操验证：旧含凭据文件所在仓库已转私有 + v2 bundle 无凭据 + 匿名下载实测 404）；凭据轮换建议已告知用户

### 8.3 实例操作通道（无 SSH → 自建 SSH 通道 ✅ D12）
> 约束: ModelScope Notebook 不提供原生 SSH；社区已验证可在容器内自建（OpenSSH + Tailscale，同款 AMD 实例，附一键脚本）

**通道阶梯（按优雅度排序）**：
1. **Tailscale 私有组网 → 真 SSH（主通道）**
   - 实例内跑 bootstrap：sshd + tailscale（无 /dev/net/tun 时用 userspace networking 模式）加入 tailnet
   - 本机同时加入 → 我从开发机 Shell 直接 `ssh/rsync/scp` 操作实例，原生终端全能力（含端口转发）
   - 用户需要：注册 Tailscale 账号（免费 3 用户/100 设备）+ 提供 auth key（或首次授权接管一次）
   - 注意：非官方功能，实例回收后重跑幂等 bootstrap（脚本常驻 ModelScope 仓库）
2. **tmate 反向隧道（备选）**：实例内一条命令拉起公网 SSH 会话（零账号零配置），走海外中继，带宽小——适合命令/日志，不适合大文件
3. **浏览器自动化驱动 JupyterLab（后备）**：内置 Terminal + 文件上传 + cell 执行；首次登录用户接管一次

**大文件传输原则**：代码/数据/结果一律走 **ModelScope 仓库通道**（`modelscope upload/download`，用户 Access Token），不依赖隧道带宽；副产品 = 解决实例存储不持久风险

**执行模型**: 长训一律 `tmux`/`nohup` 在实例内跑，连接通道只用于装环境、点火、观察；任何通道断开不影响训练

### 8.4 ModelScope NVIDIA A10 实例（不使用）
- D10: 全部实验只在 AMD 实例执行；A10 36h 额度仅作应急储备，不主动消耗

## 9. 风险登记册

| 风险 | 等级 | 对策 |
|---|---|---|
| **1h 无操作自动关机**打断长批次（单批最长 ~8.4h） | 高 | W1-V3 实测存活；被关则 agent 侧 Schedule 每 45min SSH 心跳保活 + 分批跑法；setsid 只防 SSH 断连不防平台关机 |
| **额度不足**（倍率若 ~1.35×，最小集需 ~108h > 余额） | 高 | W1-V4 锁倍率 + V1 读实际额度后立即复核；P0b 注册前置；压缩顺序见 §4 |
| **出境网络**（Tailscale 协调服务器不可达→已绕过、GitHub HTTP2 framing、Berkeley 源极慢） | 已发生 | SSH 已打通（Tailscale userspace）；git HTTP/1.1；数据走 hf-mirror；重大下载本机中转 scp |
| **numpy2/mujoco-py ABI 重编依赖系统盘状态**（/root/.cache 清空后需重编；gcc 链路未入恢复脚本） | 中（已发生 1 次） | install_amd.sh（W1 整合）加入检测+重编步骤；cymj .so 本体在持久盘 site-packages，通常无需重编 |
| **antmaze/Adroit/Pixels 路径在 numpy2 组合下从未运行**（P3/P4/P5） | 中 | 各阶段 W1 先冒烟再跑批（§5 已注明）；失败则该阶段单独排查，不阻塞其他阶段 |
| fleet 稳态倍率 ≈1.33× | 中（待 V4） | 倍率锁死后重排批次与额度；主机侧瓶颈诊断（V4 的 rocm-smi 对比）若确认可调 -P 与批量搬运策略 |
| 激励计划审核不通过 / 到账慢 | 中 | NB 严格满足审核维度；P0 后尽早提交排期 |
| ModelScope 免费实例存储不持久 | 低（持久盘已验证） | 资产三层备份（持久盘 + ModelScope 私有仓库 + 本地）；每阶段收尾同步 |
| wandb 数据可靠性（自研 .wandb 解析丢字段） | 低 | 双源对账（wandb sync 云端导出 vs 自研解析）+ 41/51 条完成度硬检查 |
| SACfD/Online 无现成脚本 | 低 | train_baselines.py ≤50 行 + 1k 步冒烟验收（P2_SPEC D1） |

## 10. 决策日志

| # | 决策 | 状态 |
|---|---|---|
| D1 | 技术路线: 官方 JAX 仓库直接复现 | ✅ 已定 |
| D2 | 首个套件: D4RL Locomotion | ✅ 已定 |
| D3 | wandb: **offline 训练 + sync 校验**（v17 修订；实例网络下 offline 实测可靠） | ✅ 已修订 |
| D4 | G1 校准策略: 先小步短程三方实测吞吐定分工 | ✅ 已定 |
| D5 | 云平台: ModelScope AMD 实例（100h, 200GB RAM, 192GB 显存）主力 + NVIDIA A10（36h）备胎 | ✅ 已定 |
| D6 | DrQ-v2 基线: 引用论文数值，不自己跑 | ✅ 已定 |
| D7 | 实验规模: **方案 B 简历级 = 验收门最小结果集 62 runs / ~113 run·h**（v22 统一口径） | ✅ 已修订 |
| D8 | Pixels 不降级，上云端（AMD 200GB 内存完全满足） | ✅ 已定 |
| D9 | 参与 AMD x 魔搭激励计划扩额（注册 +100h，NB×3 + 文×2 潜在 +275h） | ✅ 已定 |
| D10 | **全部实验只在 AMD 实例执行**，A10/本机不跑任何实验（A10 36h 留作应急） | ✅ 已定 |
| D11 | **AMD 实例内 fleet 并发**（xargs 模式；并发数与倍率待 W1-V4 锁定，表观 2.53× 已证不可信） | ✅ 修订中 |
| D12 | **实例操作通道**: Tailscale userspace SSH ✅ 已打通（v15；tmate/浏览器为后备）；computer use 仅用于开关实例 | ✅ 已打通 |

## 11. 变更记录

- v1 (2026-10-17): 初版。含资源盘点、算力预算、分阶段计划、决策日志；待定 D7。
- v2 (2026-10-17): D7 定为方案 B；接入 AMD 实例（100h/200GB/192GB）为主力；新增激励计划参与策略（2.4 节，复现产物映射为 NB/文章）；P0 拆分为 P0a/b/c（本机 NVIDIA 栈、AMD ROCm 兼容性专项、注册与内容提交）；新增 JAX-ROCm 兼容性风险与降级阶梯。
- v3 (2026-10-17): 完成 JAX-ROCm 适配调研并写入 8.2 节：官方 wheel 存在（jaxlib + jax-rocm{N}-plugin），版本矩阵确认（0.4.26+ 起支持 MI300X），AMD 环境需 py3.10+；确定 R1/R2/R3 三级适配路线；确认 tfp 依赖面极小可自实现兜底。
- v4 (2026-10-17): 镜像确认 `ubuntu22.04-rocm7.2.3-py312-torch2.12.0` → R2 路线锁定（jax 0.8.x + rocm7 plugin）；决策改用实例内自建 conda py3.10（规避 d4rl/mujoco-py 与 py3.12 兼容风险）；新增 AMD 环境安装单与适配改动清单（tfp 自实现、API 平移、requirements-rocm.txt）。
- v5 (2026-10-17): D10 — 弃用本机训练，全部实验上云；本机转纯开发机；A10 角色改为「原版栈对拍参考 + 兜底」；P0a 改为 A10 建参考栈（~1h）；P1 改为 AMD/A10 双云校准；额度账无冗余 → 新增额度节奏管控对策。
- v6 (2026-10-17): 引入 **fleet 并发调度**作为核心吞吐手段——单 run 为 Python 调度瓶颈（GPU 占用 <10%），实例内 6~7 路并发（像素 2~3 路）将额度有效价值放大约 ×5，方案 B 500 run·h 预计仅需 ~100 实例在线小时；技术路线新增 fleet 调度器；预算表改为 MI300X 口径并注明并发换算；P1 增加并发扩展曲线实测；新增 todos。
- v7 (2026-10-17): 按 D10/D11 收敛资源与执行口径——**只跑 AMD 单实例**（A10 与本机均不执行实验，A10 留作应急）；fleet 并发保留为核心手段；删除 A10 参考栈阶段（P0a 直接为 AMD 环境）；tfp 对拍改走 CPU 临时 venv（不占 GPU 额度）；预算与风险表同步收敛。
- v8 (2026-10-17): 完成适配面静态排查：确认 `jax.tree_map`（dataset.py）1 处必改、`FrozenDict.copy` 待实测、PRNGKey 兼容；新增 mujoco-py apt 系统库清单、EGL/OSMesa 渲染兜底、评估口径（训练中 10 eps / 终评 100 trials）、断点续训策略（checkpoint + fleet 自动补跑）。
- v9 (2026-10-17): D12 — 明确无 SSH 下的实例操作通道：agent 浏览器自动化驱动 JupyterLab（Terminal/文件上传/cell），ModelScope 仓库（用户 Access Token）作双向文件通道并顺带解决存储不持久；长训 tmux 化；纯 cell 执行为退路；新增对应风险条目。
- v10 (2026-10-17): D12 升级——发现社区同款实例（免费 AMD Notebook）OpenSSH+Tailscale 自建 SSH 成功实践（含一键脚本），通道改为三级阶梯：Tailscale 真 SSH（主）→ tmate（备）→ 浏览器驱动 JupyterLab（后备）；大文件固定走 ModelScope 仓库通道。
- v11 (2026-10-17): **适配开发完成**——`rocm-adapt` 分支提交（4 文件，算法零改动）；对拍单测 4/4 通过（log_prob 与 tfp 差 ≈1e-6）；deployment 包就绪（setup_amd.sh / smoke_test.py 8 项 / requirements-rocm.txt / .secrets.env）。经验记录: jaxlib 0.3.x 已从 PyPI 下架（GCS 归档源可装）、flax↔jax config API 配对敏感、scipy≥1.13 移除 tril 会挂老 jax。待办: fleet 调度器、SACfD 脚本、聚合脚本（P1/P2 前完成即可）。
- v12 (2026-10-17): 持久化策略落地（用户提醒 notebook 重启清系统盘）——conda/数据集/Tailscale 状态全落持久盘（自动探测挂载点），$HOME 仅符号链接；重启恢复两级（setup_amd.sh 增量 / env.sh 秒级）；冒烟新增 9-persist-disk；数据集容量与持久盘对账列入 P0a。
- v13 (2026-10-17): **P0a 实战完成 ~90%**——实例实况：23 核/200GB/NFS 持久盘 /mnt/data（=/mnt/workspace）、ROCm 7.2.3；**JAX_DEVICES [RocmDevice(id=0)] 验证通过**（jax 0.9.2 + jax-rocm7-plugin 0.9.2 PyPI 四件套，无需 wheelhouse）；mujoco-py 编译成功。**环境矩阵修正**：训练环境锁定 py3.11（jax 0.9+ 需 py≥3.11 ∩ mujoco-py 需 py≤3.11，交集唯一）；py3.12 挂 distutils、py3.10 无 ROCm wheel。通道结论：Tailscale ✗（协调服务器国内不可达）→ 改 tmate 备选；网页终端可用但偶发冻结（刷新可恢复）；命令通道 = 粘贴/字符级 keydown + base64 脚本（免疫引号转义）。d4rl 待装（GitHub clone HTTP2 抽风，已切 HTTP/1.1）。全部资产已存持久盘 + v2 bundle（无凭据）备份至私有仓库 CCLLUU/rlpd-repro。恢复手册：deployment/RESUME.md。凭据暴露事件（~1 分钟）已复盘，建议轮换。
- v14 (2026-10-17): **按第三方核查报告（PLAN_VERIFICATION_REPORT.md）全面整改**——一手复核确认 A1/A2/A3/A4/B2/B3/B4/F3/F4/F5 属实（F6 为措辞问题）。修正：① §8.2 重写——py3.11 决策补充 d4rl 版本依据（git master 有 `python_requires<3.11` 拒装，**改用 PyPI d4rl==1.1**：无上限、无 TF/mjrl 依赖，报告结论方向成立但有更简出路）；jaxlib 0.8.x GitHub wheel URL 404 与 cp310 写法错误改正为 **PyPI 四件套同版本 pin（jax==0.9.2 实测通过）**；删除 tensorflow-cpu（d4rl 走 h5py，核查 B4）；补 dmcgym（train_finetuning 顶层 import 曾漏）。② 适配清单计数修正为"代码文件 4 个"；对拍诚实化（核查 F4）：恢复 2 个失败用例为 xfail 并注明原因，新增走 sample() API 的统计矩用例，现状 **5 passed + 1 xfailed**，等价性覆盖范围与尾部行为差异如实记录。③ 冒烟修复（核查 B3）：sys.path 缺陷修正、item 8 改可选 SKIP、新增 item 10 train_finetuning 模块导入，G0a 判据=10 项全 PASS + 1k 步实跑。④ checkpoint 落点（核查 B2）：训练显式传 `--log_dir=/mnt/data/ckpt/<env>/s<seed>`，fleet 每 run 独立 log_dir 列为验收项。⑤ §5 验收门全部补可判据（核查 F5）。⑥ 计数修正（核查 F3）。⑦ 安全规则成文：bundle 永不含 .secrets.env。交付物同步更新：requirements-rocm.txt 重写、smoke_test.py 10 项、install_step2.sh、RESUME.md、test_tanh_normal.py（5 passed + 1 xfailed 实测）。
- v15 (2026-10-17): **G0a 通过，P0a 收口**——冒烟终态 9 PASS + 1 SKIP；SSH（Tailscale userspace）打通后完成 G0a 实跑两轮：① 默认参数 1k 步（16s 正常退出，eval/return=110.3，仅随机采动）；② `--start_training=0` 1k 步（28s，critic_loss 51.5→2.5、q 0.5→43.0、temperature 1.0→0.75、entropy −2.3→3.68、eval/return=196.7、training/return=−243.4）——**agent.update 在线路径实测健康**。重要判读：`max_steps < start_training(默认 10000)` 时训练循环不触发 update、stdout 近乎无输出且正常退出，勿误判为崩溃；wandb 离线指标可从 .wandb protobuf 直读（HistoryItem=`\x12<len>key \x82\x01<len>value_json`）。下一步 P1 吞吐校准（5k 步 + 并发曲线 1/2/4/6 路）。
- v16 (2026-10-17): **G1 通过，P1 收口，预算表实测修订**——① 单路吞吐：UTD1（sac config）5000 步/29s ≈ 2ms/step；UTD20+论文 config（E=10, LN）2000 步/_runtime=128s ≈ 33ms/step，单梯度步 ~1.7ms 与 UTD1 一致（吞吐线性于 UTD，GPU 非单 run 瓶颈）。② 并发曲线（2k 步论文口径×6 run 组）：1/2/4/6 路 = 168/193/334/398s → 总吞吐 1.00/1.74/2.01/2.53×，fleet 并发数定为 6（xargs -P 6 模式，无需调度器）。③ 预算修订：Locomotion 2.4h、AntMaze 3.5h、Adroit 9.5h、SACfD 0.2~0.6h；方案 B 总量 ≈400 run·h → **~160 实例小时**（原估 100h 偏乐观；100h 现有额度覆盖主结果大头，P0b +100h 后覆盖全方案 B）。fleet 调度器从交付物清单降级为 xargs 模式（极简原则）。
- v17 (2026-10-17): **按第二轮第三方核查（26 条）全面整改**——复核确认关键指控：① A1 属实（Fig.19=L554 'Ablation on Expert Data' 被误当主图；主结果=Fig.4 12 任务 250k）；② A2 属实（SACfD=plain SAC+seeded，论文 §5 L233 原文，非'RLPD 全配置换 buffer'）；③ B5 属实（2k 步基准非稳态开销 ~50%，稳态倍率 ≈1.33× 而非 2.53×，'GPU 喂饱'系无 rocm-smi 证据的推测，瓶颈疑为主机侧）。整改：P2 重新定位（Fig.19 三任务 200k + walker2d-medium-replay 谱系代表 250k，4 任务 × 3 算法 × 3 seeds = 36 runs，规格见 P2_SPEC.md v2）；SACfD 重定义（configs/sac_config + seeded 固定容量循环覆盖 buffer + UTD20）；G2 判据改 D4RL REF 绝对线（≥90）+ 如实报告制 + 反向结果 checklist；预算重算为验收门最小结果集 ~76 runs ≈ 146 run·h → 36~108 实例小时（倍率真值 W1-V4 锁定），额度覆盖结论推迟；W1 新增 V1 额度读数 / V2 ckpt save→restore 往返 / V3 关机存活 / V4 倍率重测 / V5 数据可比性 五项验证；P2_SPEC 修复 D1 buffer 语义（固定容量非 ds_size+max_steps）、D3 补 cwd、D4 restore_checkpoint 正名、D5 判定源改 final_eval.json + 双源对账；同步 §2.2/§3/§4/§5/§6/§7/§8.1/§8.2/§9/决策日志（D3/D7/D11/D12）；本地清理 bootstrap_ts_filled.sh、.gitignore、旧安装脚本 DEPRECATED 标记（W1 上实例执行）。
- v18 (2026-10-17): **P3~P7 规格书定稿（P3_P7_SPEC.md）**——① P3 收敛为 antmaze-umaze-v2 ×6 runs（CDQ-off 复用主结果，21 run·h），确认 reward−1 内置与 100-trials 终评系论文明确规定；② P4 摸清依赖链：mj_envs(philipjball fork+submodule) + AWAC .npy 两类文件（pen2_sparse.npy/pen_bc_sparse4.npy，源码 L72/L86），数据源 Google Drive→hf-mirror 兜底；③ P5 数据源改 conglu/vd4rl HF 官方镜像（Google Drive 作废），先 5k 校准 run，Online 对照需检查 pixels 脚本 combine 路径（预计同崩）；④ P6 复用修正：LN-on 复用 P2 runs → 9 新 runs/18 run·h；⑤ P7 输出 11 张图表三态表（Fig.6/9 预标注未复现）+ NB2/文2；⑥ 全方案 B 总量修正 ~130 run·h → 32~96 实例小时；⑦ 执行顺序：P2→P3（可同批）→P4（依赖链早准备）→P6（依赖 P2）→P5（校准可并行）→P7。
- v19 (2026-10-17): **P3~P7 规格详细化（P3_P7_SPEC.md v2，217 行）+ d4rl 1.1 源码级核查**——PyPI wheel 解包核对：① 数据本地文件名 = URL basename（filepath_from_url），支持 D4RL_DATASET_DIR 覆盖；② **antmaze-umaze-v2 实参：max_episode_steps=700、REF 0/1（成功率直通）、v2_resets=True**；③ **walker2d-medium-replay 本地名 = walker_mixed.hdf5**（Berkeley 源文件名 walker_mixed.hdf5）；④ hf-mirror imone/D4RL 扁平结构实测（HF API 全清单），**旧 download_data.sh 的 data/ 前缀路径全错会 404，已修正**（5 文件映射：expert×2、walker_mixed、walker2d_medium、antmaze v2 同名）；⑤ **AWAC .npy 不在 imone/D4RL**（P4 数据缺口）：三路方案（Google Drive 本机 scp / HF 再搜 / pen hdf5 自转 rlkit 格式兜底，源文件已确认在）；⑥ pen-binary-v0 env 由 mj_envs 提供（d4rl 1.1 无 adroit 注册）；⑦ pixels 脚本 iterator+combine 双队列结构确认 → offline_ratio=0 同崩 → train_baselines_pixels.py（~40 行）规格入册；⑧ binary_datasets 源码确认（pen2_sparse.npy+pen_bc_sparse4.npy、0.9 split、remove_terminals）；⑨ P3_P7_SPEC v2 各阶段补 W1 验证清单/命令模板/风险表；P3 时长修正（700 步 episode）。
- v20 (2026-10-17): **规格文档拆分为独立文件**（用户要求与 P2_SPEC 对称）——P3_P7_SPEC.md 拆为 P3_SPEC.md / P4_SPEC.md / P5_SPEC.md / P6_SPEC.md / P7_SPEC.md（各 63~68 行，自包含：事实基础/命令模板/W1 验证清单/风险表/验收），§5 引用同步；upload_all.sh 同步新文件名。
- v21 (2026-10-17): **按第三轮核查（30 条 + 1 条撤回）整改**——一手验证证实：① **d4rl get_normalized_score 返回 0~1 不乘 100**（offline_env.py L71-74；final_eval.py 已 ×100 实现 ✓，规格补约定；aggregate 判据 2 原用原始 return 恒 PASS → v3 重写：内置 REF 表归一化后判 50k≥60 分）；② **cheetah-run PLANET action_repeat=4 非 2**（L58-61）→ 命令显式 --action_repeat=2 对齐论文 Table 1；③ vd4rl 目录 **cheetah_run 下划线**（连字符读不到）；④ start_training 陷阱三处复发全修（P4 冒烟/P5 校准改 --start_training=0）；⑤ **SACfD 重定为 UTD=1**（论文 §4.5 'without additional design decisions'——高 UTD 属 §4.3 Green 核心；P2 预算 73→26 run·h）；⑥ P3 CDQ 消融域改 **antmaze-large-diverse-v2**（Fig.8 原域，umaze 无信号）9 runs/30 run·h；⑦ P6 claim 修正：LN 消融复现对象 = **Fig.19 谱系**（Fig.2/7 域如实标未复现），Seeded 判据改样本效率+std 口径（Fig.11 原主张=终值相当）；⑧ P7 三态表重写（Fig.5→P5、Fig.14→P3、Fig.19→claim 3、删错误映射）；⑨ 额度统一 62 runs ≈ 89 run·h → **56~66 实例小时**（删 4× 上界；现有 100h 可覆盖）；⑩ 代码修正：aggregate v3（REF 表/BENCH 读 P2_BENCHMARK/glob 含 exp_prefix 层/g2_judge 实装）、data_sanity ≥95（REF=数据集最强 episode，≥100 必误报）、download_data.sh ~/.d4rl symlink 断言、gen_runs v3（--suite p2~p6 全阶段统一，P6 Seeded 显式 --config=rlpd_config.py 防命名污染）；⑪ P4 收缩：G4=冒烟（结果级弹性，路径③稠密/二值奖励混批尺度问题）、pen-binary 无 get_normalized_score（mj_envs env，return 恰=成功占比）；⑫ §1 回写 Claim×阶段覆盖矩阵；⑬ NB2 Gallery 无 ROCm 环境约束 → cell 分层设计；⑭ 撤回条目（num_min_qs 索引语义）核实无误无需回改。
- v22 (2026-10-17): **按第四轮核查（31 条）整改**——一手验证证实 4 个 run 起不来的硬 bug 并全部修复：① gen_runs suite_p3 的 --num_min_qs=1 是不存在的顶层 flag（应 --config.num_min_qs）→ 6 个 large-diverse run 必 FATAL，已修；② --checkpoint_model 追加给像素脚本（无此 flag，用 --save_dir）→ p5 全 FATAL，cmd() 分流已修且补 --save_dir；③ gen_runs 引用未交付的 train_baselines_pixels.py → **文件已写出**（Online 基线，--offline_ratio=0 对齐 batch 256）；④ --phase final_eval 恢复（p2_final_eval 36 行，exp_prefix 层正确）。预算修正：P2 重算**含 Online 组 = ~50 run·h**（v21 的 26 漏加 Online），全方案 B 62 runs ≈ **113 run·h → 71~84 实例小时**（1.6~1.35×）+W1/W3/余量逼近 100h → **P0b +100h 为必需项**；plan §4/§5/P2_SPEC/P7 四处数字统一。判据修复：aggregate v4（--suite 参数化 p2/p3/p5/p6、REF 补 walker2d-medium-v0/antmaze 两键、单段判定、load_benchmark 改读 JSON——md 正则误命中 transitions 列）、G6 判据 2 实装（steps-to-60 + std 描述性，n=3 不作显著性检验）、final_eval 加 latest_step None 硬失败 + get_normalized_score hasattr 分支（pen-binary/像素无此方法）、G2 判据 1 的 90 分诚实化为自设工程线（双锚定：数据集上限 100 + Fig.19 读图）、V5 阈值统一 ≥95（三处 ≥100 与凭空 +5 容差清除）。数据/路径：download_data 补 large-diverse 文件、~/.d4rl 三处统一 /mnt/workspace/.d4rl 且断言升级为检查链接目标、env.sh 同步；pixels action_repeat 渲染方向修正（env-steps 固定时与 ar 无关）。文档扫尾：P3（9 runs/30 run·h、large-diverse 数据+1000 步+方差带风险）、P4（header/风险表/预灌注释/hand_manipulation_suite 澄清/submodule pin）、P5（cheetah_run/V5-b 10k/ar 方向）、P2（P2.1 header/50 run·h/≥95）、§1 claim 2/3 正文对齐覆盖矩阵、§7 Adroit 兜底与 V-D4RL 下划线。
- v23 (2026-10-17): **按第五轮核查（31 条，含 1 条撤回确认）整改**——4 个代码级缺陷全修：① cmd() 无差别追加训练 flag（final_eval 无 checkpoint_model/project_name）→ **按脚本能力分流构造**（final_eval 零追加/像素走 --save_dir/其余加 checkpoint_model），W3 终评 36 条恢复可跑；② train_baselines_pixels 补漏抄的 --replay_buffer_size flag（缺则启动即 AttributeError）；③ 同文件补 memory_efficient 分支的 pack_obs_and_next_obs=True（缺则 drq._unpack UnboundLocalError）；④ 自检实装：期望行数断言 + bash -n 全行 + 生成即跑（SUITES 期望值此前被丢弃）。判据链补全：⑤ G6 判据 2 steps-to-60 实装（curves_check 输出首次 ≥60 步数）；⑥ aggregate table() 的 normalized_score=None 回退 return_raw（Adroit/像素合法 None 不再 TypeError）；⑦ 单段判定（占位 body 替换法弃用，改具名组装）；⑧ load_benchmark 缺文件打印警告；⑨ **p3_final_eval/p6_final_eval suite 新增**（antmaze 100 trials 论文规定口径；NoLN exp_prefix 无 _LN 已处理）。监控与传输：⑩ status.sh 参数化（logs/pN_*.log + 双层 final_eval glob + ckpt 体量 + 持久盘余量）；⑪ upload_all 补 env.sh/requirements-rocm.txt/smoke_test.py（env.sh 曾漏传致实例侧 ~/.d4rl 指回 /mnt/data）且远端校验升级为 ast.parse + **每脚本 --help 抓 unknown flag** + bash -n + gen_runs dry-run。像素磁盘：⑫ save_checkpoint 无 keep 问题记录（pixels 侧 ckpt 体量入 status.sh 的 du 输出与 P5 对账）。文档：⑬ ≥100 残留清除（data_sanity docstring/P2 D2）、P2 §8 18~54→31~37、plan §4 1.35~4× 残留清除、判据 1 锚定统一（工程线主锚=数据集上限 100，读图仅佐证）、六份 spec 上游指 v22；⑭ **根目录 git init**（.gitignore 生效）+ rlpd 独立仓库提交 3 个新脚本（P7 收尾项提前）。确认无需再议项与上轮一致。
