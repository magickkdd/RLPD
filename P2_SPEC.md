# P2 规格书 v2：Locomotion 结果 + SACfD/Online 对照

> 版本: 2026-10-17 v2.1（第四轮核查扫尾）· 前置: P0a(G0a ✅) / P1(G1 ✅ 带 V4 复测尾巴) · 上游: REPRODUCTION_PLAN.md v22 · 场地: AMD MI300X 实例

## 0. 定位（v2 修正）

**P2 复现对象 = Fig.19（"D4RL Ablation on Expert Data"，3 expert 任务，0~200k）+ Fig.4 语义的一个低质量数据代表点**。

- 论文 locomotion **主结果**是 Fig.4（12 任务 × 250k，对照 Prior SoTA=Off2On 与 SACfD）；方案 B 预算下不全量复现 Fig.4，以其谱系代表任务（medium-replay）+ Fig.19 全三任务覆盖主结论的关键面：**离线数据加速 + LayerNorm/ensemble 带来的稳定性**
- claim 覆盖对账：claim 1（超越 Prior SoTA）→ P2 用 Off2On 论文读图值作参考线（非正式对账）；claim 2（对称采样优于 seeded）→ **P6**（Fig.10/11）；claim 3（像素）→ P5
- 论文 A.3 预期管理（必须写进 NB2）："locomotion 任务不是 offline data 的最佳用例，纯在线也能较快解决"——expert 任务上 Online 基线有竞争力是**论文自认**的现象，不是复现失败

## 1. 事实基础（v2 修订，全部带出处）

| # | 事实 | 来源 |
|---|---|---|
| F1 | **Fig.19** = hopper/walker2d/halfcheetah **-expert-v0** × **0~200k**，legend = BC / REDQ(Online) / no LN / RLPD；**Fig.4** = 12 locomotion 任务 × **0~250k**，对照 Prior SoTA(Off2On) 与 SACfD | RLPD_paper.md L554-558（Fig.19 图注）、L219-231（Fig.4 图注与 §5） |
| F2 | **SACfD 定义（论文 §5 L233）** = plain SAC（**无 4.5 节设计决策——§4.3 的高 UTD 属 Green 核心设计，故 SACfD 为 UTD=1**）+ "initializes the online replay buffer with the offline data"。config = `configs/sac_config.py`（num_qs=2、无 LN、2 层）+ seeded buffer + **UTD=1**（v21 修正；与 RLPD 的差异 = UTD+架构+buffer 三者，论文原口径如此；变量隔离由 P6 Seeded(RLPD-config) 补全） | RLPD_paper.md L233 + §4.3/4.5 结构 |
| F3 | RLPD 超参 = `configs/rlpd_config.py`（E=10、M=2 CDQ、LN=true）+ UTD20 + batch 128+128 + lr 3e-4 + γ0.99 + ρ0.005 + target_entropy=−dim(A)/2（代码默认即此，sac_learner.py L79） | 论文 Table 1/2 + rlpd 源码 |
| F4 | d4rl 1.1 locomotion env id 仅 **-v0** 注册；get_normalized_score 的 REF 分数为 v0 口径（halfcheetah-expert: −280.179 / 12135.0）；**在位数据文件为 v2 采集数据改名喂 v0 env**（halfcheetah 已验证 h5 可读 + 形状，**未验证 v0/v2 数据分布可比性**——W1 补 best-in-dataset sanity 检查并写入口径说明） | d4rl 1.1 源码 + RESUME.md |
| F5 | **并发倍率 v2 修正**：2k 步基准中非稳态开销（import ~40s + E10/UTD20 jit 编译 ~60s + i=0 eval 10k env-steps ~10s）占比 ~50%；6 路表观 2.53× 扣除同额开销后**稳态倍率 ≈1.33×（区间 1.3~1.6×）**。但 MI300X 算力远高于 30 env-steps/s × UTD20 的需求（每秒仅数 GFLOPs）→ 瓶颈极可能在**主机侧**（Python 循环 / combine() 交织拷贝 / kernel launch / H2D），多进程并发理论上可扩展性好——**两种瓶颈诊断的对策相反，W1-V4 重测裁定** | 本会话 P1 数据反推 + 核查报告 B5/B7 |
| F6 | **归一化约定（v21 统一）**：`env.get_normalized_score()` 返回 **0~1**（d4rl offline_env.py L71-74 实测，不乘 100）；**入库/对账一律 ×100 对齐论文 0~100 口径**（final_eval.py 已实现；aggregate/判据/V3-3 均按 0~100）。训练中 eval = 10 eps（默认），终评 100 eps 系扩展口径（论文 B.1 仅规定 AntMaze/Adroit 100 trials）——NB2 注明 | d4rl 源码实测 + 论文 B.1 |
| F7 | `--config` 相对路径 + `from configs import sac_config` → **所有 run 必须 cwd=/mnt/workspace/rlpd**；wandb offline 目录落在 cwd | train_finetuning.py L58-63 |
| F8 | 保存 ckpt 的代码被 `try/except: print(...)` 静默包裹（train_finetuning.py L212-218），且 G0a 从未开过 `--checkpoint_model`——**save→restore 往返必须 W1 显式验证**（正确 API = `flax.training.checkpoints.restore_checkpoint`，无 load） | rlpd 源码 |
| F9 | `offline_ratio=0/1` 会在 `combine()` 触发 broadcast ValueError（实测）→ **Online 基线不能用 flag 实现**，必须走 train_baselines.py 独立采样路径 | 核查报告 C24 实测引用 |

**基准值口径（v2 重设，替代"读图 ±10 分"）**：
- **主判据用 D4RL REF 绝对线**（可追溯、不依赖读图）：halfcheetah-expert-v0 3-seed 终值归一化分 ≥ 90
- Fig.19 人工读图值记入 P2_BENCHMARK.json（读图误差 ±5）：**不作为论文数值对账依据**；判据 1 的工程线 90 见 §6.1 锚定说明
- 加一条 sanity：W1 时用各数据集 best episode return 过 get_normalized_score，expert 数据应 **≥ 95**（REF 本身取自数据集最强 episode → best_norm 天然落 97~100；≥100 会必误报——v21/v22 修正），结果写入 P2_BENCHMARK.json

## 2. 任务矩阵与预算（v2）

| 组 | 脚本/config | 任务 | seeds | runs | 单 run |
|---|---|---|---|---|---|
| RLPD | train_finetuning.py + rlpd_config | 3×expert-v0 | 42/43/44 | 9 | ~1.9h |
| RLPD | 同上 | walker2d-medium-replay-v0 | 42/43/44 | 3 | ~2.4h |
| SACfD | train_baselines.py --baseline=sacfd + **sac_config**（plain SAC + seeded + **UTD=1**，F2 v21） | 4 任务同上 | 42/43/44 | 12 | **~0.15h**（2ms/step；单梯度步慢于 RLPD 但 UTD1 便宜 20 倍） |
| Online | train_baselines.py --baseline=online + rlpd_config（REDQ 无离线数据） | 4 任务同上 | 42/43/44 | 12 | ~1.9-2.4h |

- BC：不跑，Fig.19 读图值作参考线（NB2 注明）
- 合计 **36 runs ≈ 50 run·h**（v22 重算含 Online 组：RLPD 24.3 + SACfD 1.8 + Online ~24；此前 26 漏加 Online）；倍率 1.35~1.6×（V4 锁定）→ **~31~37 实例小时**
- 统一 flag：`--utd_ratio=20 --start_training=5000 --max_steps={200000|250000} --config=... --seed=... --checkpoint_model=True`；**每条命令以 `cd /mnt/workspace/rlpd && ` 开头**（F7）

## 3. 交付物规格（v2）

### D1. `rlpd/train_baselines.py`（新增，派生自 train_finetuning.py，改动 ≤50 行）

```
flag: --baseline ∈ {sacfd, online}
sacfd:
  1) ReplayBuffer 容量 = ds_size（固定容量循环覆盖，在线数据按 FIFO 逐渐稀释离线数据——论文 "initialize" 语义；
     不用 ds_size+max_steps：那会使离线数据恒占 83%，变成固定混合，偏离两个基线的语义（核查 C11）
  2) 预灌：遍历 ds 逐条 insert，print 计时（预估 30-60s，计入预算与冒烟）
  3) 在线采样：batch = replay_buffer.sample(batch_size*utd)（删除对称采样与 combine）
  4) config 默认指向 configs/sac_config.py（plain SAC，F2）
online:
  1) 不构建 D4RLDataset；buffer 初始为空；采样同 sacfd（F9：不能用 offline_ratio=0 实现）
  2) config 保持 rlpd_config（Online 对照 = REDQ 结构，与 RLPD 唯一差异 = 无离线数据）
```

**验收**：两模式各跑 1k 步（halfcheetah-expert-v0）；sacfd 日志显示预灌条数与耗时；critic_loss 有限；**同时验证 --checkpoint_model 的 save→restore 往返**（F8）

### D2. 数据准备（~/.d4rl/datasets/，v2 增可比性检查）

| 目标文件名 | hf-mirror 来源 | 校验 |
|---|---|---|
| hopper_expert.hdf5 | imone/D4RL data/hopper/hopper_expert-v2.hdf5 | h5 直读 + best-in-dataset norm ≥ 95 |
| walker2d_expert.hdf5 | data/walker2d/walker2d_expert-v2.hdf5 | 同上 |
| **walker_mixed.hdf5**（= URL basename，v21 修正） | walker2d_medium_replay-v2.hdf5（hf-mirror 根目录，扁平） | h5 直读 + best-in-dataset 记录 |
| halfcheetah_expert.hdf5 | 已在位 ✅ | 补 best-in-dataset 记录 |

- best-in-dataset sanity 写入 P2_BENCHMARK.md（F4 的 v0/v2 可比性证据）；下载主路径 hf-mirror 直下，兜底本机 scp

### D3. `deployment/gen_runs.py`

- 输出行模板：`cd /mnt/workspace/rlpd && LD_LIBRARY_PATH=... XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline /mnt/data/miniforge3/envs/rlpd311/bin/python train_{finetuning,baselines}.py --env_name=... --utd_ratio=20 --start_training=5000 --max_steps=... --config=configs/{rlpd,sac}_config.py --baseline=... --seed=... --checkpoint_model=True --project_name=rlpd_p2 --log_dir=/mnt/workspace/ckpt/p2/{algo}/{env}/s{seed} > /mnt/workspace/logs/p2_{algo}_{env}_s{seed}.log 2>&1`
- 自检：36 行；`--dry-run` + `bash -n`。**gen_runs.py v4**：--suite p2/p2_final_eval/p3/p4/p5/p6（非像素脚本追加 --checkpoint_model，像素用 --save_dir；P3 用 --config.num_min_qs——v22 修 4 处硬 bug）

### D4. 终评（`rlpd/final_eval.py`，~50 行）

- `flax.training.checkpoints.restore_checkpoint`（**无 load API**，F8）加载最新 ckpt → `evaluate(agent, env, num_episodes=100)` → json 写 **`{ckpt_dir 的父目录}`/final_eval.json = `{log_dir}/{exp_prefix}/final_eval.json`**（ckpt 真实路径含 exp_prefix 一层：`{log_dir}/{s<seed>_0pretrain[_LN]}/checkpoints`，train_finetuning.py L93-96；aggregate glob 已按此匹配）
- W1-V2 先验证往返，失败则触发备选口径：训练末 5 次 eval（50 eps）均值，NB2 注明
- 100 eps 口径在 P2_RESULTS.md/NB2 注明"论文 locomotion 未规定终评条数，取 100 系扩展口径"（F6）

### D5. `deployment/aggregate.py`（v2 修口径）

- **判定数据源 = final_eval.json（100 eps）**；训练中 evaluation/return 曲线（每 run 期望 **41 条** = 200k/5000+1，或 51 条 = 250k/5000+1，完成度检查以此为准）仅用于判据 2 的形状与 50k 步内达标检查
- 自研 .wandb protobuf 解析为主源；**W2 期间对 ≥3 个 run 用 `wandb sync` 上传后云端导出对账一次**（双源防静默丢字段，核查 C14）
- 输出 P2_RESULTS.md：对账表（RLPD/SACfD/Online × 4 env：mean±std、论文参考线）、G2 逐条判定、36 runs 健康表（完成度/NaN/异常标红）

### D6. `deployment/status.sh`：进程数 + 日志尾部 + ckpt 数 + **rocm-smi 利用率（默认开，不再可选）**

## 4. W1 验证清单（v2 扩充——先证假设，再开跑批）

| # | 验证 | 方法 | 通过标准 | 裁定 |
|---|---|---|---|---|
| V1 | 额度读数 | 控制台读实际剩余小时 | 写入 RESUME.md | — |
| V2 | ckpt 往返 | sacfd 冒烟 run 开 --checkpoint_model，restore 加载并 evaluate 1 eps | final_eval.json 产出 | 失败 → D4 备选口径 |
| V3 | 关机存活 | nohup 一个 >1.5h 任务后完全不操作 | 1.5h 后进程仍活 | 被关 → 启用 Schedule 每 45min SSH 心跳保活（agent 侧自动化，无需人工） |
| V4 | **倍率重测** | 单路/4 路/6 路 × 20k 步（稳态占 ~87%），单路×2 + 6 路×2 重复 | 报告 wall 与稳态两条口径倍率 + rocm-smi 证据 | 倍率 <2 → W2 改分批跑法并压缩 P4/P6 可选 run |
| V5 | 数据可比性 | D2 的 best-in-dataset sanity | expert 数据 best_norm **≥ 95** | <95 → P2_BENCHMARK.json 记录偏差并评估数据强度（无凭空容差） |

## 5. 执行流程

| 波次 | 内容 | 预算 |
|---|---|---|
| W1 | 重连 → 同步本地 deployment/ → D1/D2/D3 → V1~V5 → sacfd/online 1k 冒烟 | ~4-5h |
| W2 | 主批 36 runs；**倍率 <2 时分批**（每批 ≤6 runs 且墙钟 8.4h 内完成 + 保活）；每 2-4h status.sh 轮询 + wandb sync 抽查 | 18~54h（V4 后定） |
| W3 | 终评 36 runs（6 路并发） | ~2h |
| W4 | aggregate → P2_RESULTS.md → 同步本地 → G2 判定 | ~0.5h |

断点续跑：无 resume 能力（官方如此），失败 run 整跑重替；runs.txt 剔除已完成行后重喂 xargs。

## 6. G2 判据（v2 重设）

1. **绝对线**：halfcheetah-expert-v0 3-seed 终值归一化分（100 eps、0~100 口径）**≥ 90**——出处 = 论文 Fig.19 halfcheetah-expert 终值读图 ~100（P2_BENCHMARK.md 记档）− 10 容差，并与 data_sanity 的 best_norm（≥95）交叉锚定，非凭空阈值
2. **样本效率形状**：3 个 expert 任务 RLPD 在 50k 步内达 **60+ 分（0~100 归一化口径）**——aggregate curves_check 用 REF 表把原始 return 归一化后判定（v21 修正：不再与原始 return 比较）
3. **对照（如实报告制，不预设成立）**：
   - RLPD vs SACfD：4 任务逐个报告差距与方向；论文预期 RLPD ≥ SACfD（Fig.4），expert 任务差距可能小（A.3 预期管理）
   - RLPD vs Online：walker2d-medium-replay 上 RLPD 应显著优于 Online（离线数据价值主叙事）；expert 上 Online 有竞争力为论文自认现象
   - 统计口径：3-seed 均值±std 全量呈现，不做显著性检验（NB2 注明论文 10 seeds 方差带，本复现 3 seeds 系额度约束）
4. **实现正确性核验（防"误判 FAIL"）**：任一反向结果先过 checklist——seed 间方差量级、sacfd 预灌耗时日志、online 无 ds 加载证据、combine 无异常、数据 best-in-dataset sanity——通过后才允许下"方向不成立"结论

## 7. 风险与对策（v2 增补）

| 风险 | 概率 | 对策 |
|---|---|---|
| 1h 无操作自动关机中断 8.4h 批次 | 高（未验证） | W1-V3 实测；被关则 Schedule 心跳保活 + 分批跑法 |
| ckpt 静默保存失败 → W3 落空 | 中 | W1-V2 往返验证前置 |
| 稳态倍率 ≈1.33× 致 P2 超预算 | 中 | V4 裁定后按 §2 区间锁预算；>54h 则砍 Online 组到 2 seeds 并 NB2 注明 |
| SACfD plain SAC 在 walker2d-expert 发散 | 中 | 如实报告（这正是论文用 LN/ensemble 的动机）；不救援参数 |
| hopper/walker2d 数据损坏 | 中 | hf-mirror + h5 校验 + scp 三级兜底 |
| v0/v2 数据分布差异引入偏差 | 中 | V5 sanity + P2_BENCHMARK.md 口径说明 + NB2 声明 |
| 自研 .wandb 解析丢字段 | 低 | D5 双源对账 + 41/51 条完成度硬检查 |

## 8. 额度对账

- P2 = 18~54 实例小时（V4 后锁定）；W1 开机先读控制台实际余额（V1）
- 下游阶段按"验收门最小结果集"重算见 REPRODUCTION_PLAN.md §4 v21
