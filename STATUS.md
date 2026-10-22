# RLPD 复现阶段性汇报

> 2026-10-17 22:45 (Asia/Shanghai) · 场地: 魔搭 AMD MI300X 实例 (dsw-2267587)
> 总体进度: P0a ✅ → P1 ✅ → W1 验证 4.5/5 → **P2 W2 主批运行中**

## 1. 里程碑总览

| 阶段 | 状态 | 说明 |
|---|---|---|
| P0a 环境搭建 (G0a) | ✅ | ROCm 7.2.3 + JAX 0.8 适配、tfp 纯 JAX 替代 |
| P1 一致性验证 (G1) | ✅ | 对拍 1e-6 级、多任务并发验证 |
| W1 验证 V1-V5 | 4.5/5 | V2/V3/V4/V5 ✅；V1 额度读数待用户 |
| **P2 W2 主批 (36 runs)** | 🔄 | 4 路并发运行中，ETA 10/18 深夜 |
| P3 AntMaze+CDQ (G3) | 冒烟 ✅ | V3-1/2/3 全过，9 runs 排队 |
| P4 Adroit (G4) | ✅ | 冒烟门通过；结果级按规格弹性跳过 |
| P5 Pixels (G5) | 冒烟 ✅ | V5-a 通过、数据就位、校准运行中 |
| P6 消融矩阵 (G6) | 排队 | 依赖 P2 walker2d-expert runs |
| P7 报告+收尾 | 排队 | — |

## 2. W1 验证结论

- **V2 ckpt 往返** ✅：sacfd 预灌 999k 条 2.4s；restore→evaluate→final_eval.json 全通；归一化 ×100 口径
- **V3 关机存活** ✅：marker 精确落盘于 20:47:35（90min 无人工操作 + 6 训练进程存活）→ 平台自动关机风险排除；45min SSH 心跳兜底
- **V4 倍率重测** ✅（20k 步 × 单路×2 / 4路 / 6路×2）：

  | 配置 | wall (20k×N) | 聚合吞吐 | 表观倍率 |
  |---|---|---|---|
  | 单路×2 | 411s / 414s | 48.5 steps/s | 1.00 |
  | 4 路 | 1467s | **54.5 steps/s** | 1.124 |
  | 6 路×2 | 2238s / 2243s | 53.5 steps/s | 1.103 |

  结论：聚合吞吐 ~54 steps/s 封顶（6 路期间 rocm-smi GPU busy=100%、内存 25/200GB、load 7/23 核）→ **W2 裁定 -P 4**（吞吐相同、单 run 延迟更低、断电损失更小）
- **V5 数据 sanity** ✅：halfcheetah best_norm=92.9（v2 数据在 v0 REF 口径下 <95，偏差已记录，判据 1 的 90 分工程线仍可达，NB2 须声明）

## 3. 重大排障记录（本阶段）

1. **gen_runs v23 flag 段拼接 bug（最关键）**：`cmd()` 把 `--project_name/--checkpoint_model` 作为独立元素经 `&&` 连接，运行时变成独立命令段 → flag 丢失（不存 ckpt）、日志重定向错位（输出全进驱动日志）。`bash -n` 语法合法故自检未拦截。**发现于点火后 15 分钟内**（wrapper argv 的 `&& --project_name` 暴露），杀批清理后重写 `cmd()` 并在自检中增加语义断言（重定向前必须内联 ckpt flag），重新点火并逐进程验证 fd 指向与 flag 齐全。若未发现，36 runs 全部不存 ckpt，W3 终评落空。
2. **mj_envs 命名空间包遮蔽**：tarball 误打内层 package 目录 → 补打完整 clone 根 → 又因解压进 site-packages 使 `site-packages/mj_envs`（无 `__init__.py`）成为命名空间包遮蔽真包、注册代码未执行 → clone 移至 site-packages 外 + 清理残留后 `PEN_BINARY_OK 100`。
3. **AWAC 数据不可得**：GDrive 本机/实例双端不可达（出境 000）、HF 无原格式镜像 → 用 imone/D4RL 的 pen-cloned-v1.hdf5 按边界自环派生 next_obs 转换为 rlkit npy 格式，**仅作 G4 冒烟链路验证**（稠密/二值奖励混批不可用于结果级，P4_SPEC §4 路径③降级理由，NB2 注明）。
4. **hf CLI 无视 HF_ENDPOINT**：huggingface_hub 2.2.0 的 `hf download` 直连 huggingface.co（Network unreachable）→ 改 curl 直抓 hf-mirror resolve URL（300 是 'medium' 子串误匹配 medium_expert/medium_replay，真实 100 文件），100/100 npz 落盘持久盘 + symlink。
5. **pkill 自匹配**：`pkill -f "hf download"` 匹配到自身命令行导致 SSH 命令自杀（255）——模式匹配类远程命令须用 `[h]f` 方括号技巧。

## 4. 预算对账（V4 后锁定）

- P2 主批 = RLPD+Online 5100k UTD20 步 ÷ 54.5 steps/s ≈ **26h** + SACfD（UTD1，≈1h）≈ **27-28h 墙钟**（规格预估 31-37h 内）
- 全程估算（P2 主批+终评+P3 13.8h+P6 9.2h+P5 ~10h）≈ **60-65h** < 规格预算 71-84h
- **依赖用户操作**：V1 额度读数（控制台剩余小时）+ P0b 激励计划注册（+100h，必需）

## 5. 备份状态

- **GitHub（新）**：`magickkdd/rlpd-repro`（根项目：规格书/计划/流水线/文档，master）+ `magickkdd/rlpd-rocm`（rlpd 代码：main + rocm-adapt），均 private；推送后 md5 抽查确认与实例运行版本零漂移
- 本地 `/home/czx/RLPD` 仍为 single source of truth；实例持久盘为二级备份
- GitHub PAT 仅一次性使用（未写入 .git/config 或文件）；复现结束后建议撤销重建

## 5.4 范围裁剪（10/18 02:30，用户决策）

- **新范围 = 只跑论文主方法 RLPD + 部分任务**：P2 RLPD 4 任务 × 3 seeds（12 runs）+ P3 antmaze-umaze ×3 seeds；**砍掉** SACfD/Online 对照（已完成的 3 个 SACfD 保留不入账）、CDQ 消融 ×6、P6 消融 ×9、P5 Pixels ×7、P4（此前已弹性跳过）
- 剩余训练时长 **~15h**（原方案 ~60h）；实例额度 45min 时触发此决策
- 判据影响：G2-1 绝对线（halfcheetah ≥90）与 G2-2 样本效率形状照常判定；G2-3 对比组声明标注"未复现（范围裁剪）"入 NB2
- 实施细节：runs_p2.txt 替换为 12 行 RLPD 清单（全量 36 行备份于 runs_p2_full36.txt）；旧队列终止时曾出现同种子重复进程事故，已清场并由 master 重开波次（4 run 全部从 ckpt 自动续训：125k/125k/115k/125k）；终评只跑 RLPD 12 个

## 5.5 续训与自愈能力（10/18 凌晨追加，响应用户"开关实例不可避免"）

- **断点续训**：train_finetuning.py / train_baselines.py 打补丁——启动时检测 `{log_dir}/{exp}/checkpoints` 的最新 ckpt（keep=20，每 5000 步一存）自动 restore 并从该步继续（已用崩溃 run 的 ckpt_125000 做隔离冒烟：`[resume] continue from step 125000` → 干净退出 exit=0）。注：在线 buffer 的已采部分不可恢复，恢复后仅含离线预灌（NB2 声明）
- **主控守护 w2_master.sh**：每 5 分钟巡检，训练进程数为 0 且 36 个 run 未全部干净退出 → 自动重建未完成清单并以 4 路开新波次（崩溃自愈 + 波次衔接二合一，替代 w2_refeed.sh）；全部完成输出 W2_ALL_COMPLETE
- **开机自拉起**：bootstrap_ts.sh 末尾追加幂等的 master 启动——实例重启后用户只需照常跑 bootstrap，训练自动恢复
- **心跳自愈**：45min 心跳增加 master 存活检查，发现守护缺失自动重启（幂等）
- 事故记录更新：3 次 segfault（23:32-23:58，GPU 扰动窗口）的 run 将由 master 在波次衔接时自动续训，不再整跑重跑

## 5.6 关机前终态（10/18 02:35）

- 额度耗尽主动关机；RLPD 4 个在跑 run 已存最新 ckpt（130k/130k/120k/125k），3 个 SACfD 完整完成
- 实例日志快照已同步本地 instance_logs/；RESUME.md 已更新为恢复手册
- 开机恢复 = 用户跑一次 bootstrap_ts.sh，其余全自动（master 续波 + 断点续训）

## 5.7 踩坑记录（10/18 新实例恢复战役，已同步项目记忆）

1. **同 seed 双写事故**：多代波次孤儿进程互踩 → 干预后必须全量清场（master+xargs+train 三类）再单点重启，核对 /proc 真实 python 进程（排除 bash 包装假计数）
2. **启动风暴 segfault**：本台 GPU 扛不住 4 路同时 XLA 初始化（dmesg runlist 风暴可定性），守护 v3 改 90s 错峰启动；osmesa 异构任务绝不与 fleet 并存
3. **巡检正则漏报**：续训 run 的 tqdm 分母=剩余步数（65001），固定 /200001 全漏报误判"未出步"
4. **数据落位**：旧实例数据放非持久位置随实例消失；download_data.sh 自带软链断言+重下，新实例开机必须先跑
5. **apt 系统库设计上每次开机要装**（libgl1-mesa/glew/osmesa6/egl1/patchelf），conda+cymj 编译缓存在持久盘免重装
6. **pkill 自匹配**（SSH 255 真凶）、**多层引号剥解**（复杂命令一律文件化执行）、**hf CLI 无视 HF_ENDPOINT**（curl 直抓 hf-mirror）、**公钥占位符覆盖**（bootstrap 公钥直填入库）——细节见项目记忆 Lessons Learned

## 5.8 8h 实例关机前终态（10/18 10:25）

- **P2 = 4/12 完成**（halfcheetah ×3 + hopper s42，200k 干净退出）；8 个 run 带存档 ckpt（hopper s44 / walker2d s42 / s43 已至 195k，各差 ~10 分钟收尾）
- 剩余训练量 ~440k 步 ≈ 2.5h；错峰波次（v3 守护）已验证可正常续跑
- 恢复手册 RESUME.md 已更新（开机 3 步：bootstrap → apt 库 → 软链）
- 新实例战役教训已固化：STATUS §5.7 + 项目记忆 10 条 Lessons Learned

## 5.9 训练后期发散发现与抢救方案（10/18 晚，终评排障链）

- **现象**：12/12 训练完成后，final_eval 全部 12/12 segfault（TF/orbax 二次初始化段；TF 先行 import 补丁已修复该崩溃）→ 修复后终评分 = **−2.57（垃圾策略）**，与训练时 92-99 分的评估矛盾
- **排障链**：faulthandler 定位 import 行 → A/B 测试排除 wandb → 叶级对拍（36/36 路径零差异，restore 结构无罪）→ 参数范数（382→5338，已加载）→ **ckpt 步数剖面锁定真因**
- **真因（实证）**：halfcheetah s42 剖面 = ckpt@105k→97 分、@150k→**99 分**、@175k→**−3 分**、@200k→−3 分 → **训练在 150k→175k 之间发散**，ckpt 忠实记录了死策略；发散窗口与 GPU 风暴时段的续训段重合（数值级损坏嫌疑，无法完全定责）
- **抢救方案（执行中）**：① 12 runs × ckpt@50k/150k 健康筛查 ② 各取发散前最后健康 ckpt 做 100 集终评 ③ G2-1 按健康平台判定 + 发散作为偏差如实记录 ④ G2-2 形状线用早期曲线（50k 步内达 60 分）不受影响
- 终评基建修复沉淀：final_eval.py TF 先行补丁、串行终评执行器（本实例并发初始化必崩）、wandb 0.30 离线解析（锚点扫描）

## 5.10 v25 根因反转：系统性早期猝死（非 GPU 风暴/续训问题）（10/18 晚）

- **筛查矩阵推翻 §5.9 认知**：v2/v3 ckpt 筛查显示 12 run 中 11 个在 ckpt 全窗口（105k~200k）死亡；wandb 训练曲线对质 → **死亡发生在 10k~35k 步（实例 1 干净连续段，早于 segfault 窗口）**——GPU 风暴、续训链路嫌疑全部排除
- **死亡画像**：walker2d 3 seeds 冲到 78~115 分后 25k~35k 骤死恒定 ~0 分；hopper 爬坡中途死；halfcheetah s42 唯一幸存（98+ 平台，185k 死）；分数恒定 + 无 NaN
- **验尸（参数级）**：死亡 run 的 **actor 参数范数爆炸 100~180 倍**（17→1761~3117），critic 相对正常（289~1321 vs fresh 89），|log_alpha| 1.5~2.3（α 崩塌）；健康 halfcheetah s42 actor 仅 57.2
- **q 曲线（wandb 锚点提取）石锤演变链**：hopper s43 @20000 q=308/entropy=−1.2 → @20995 q=1007/entropy=+8.5e23 → entropy 指数爆到 3e24 → 即 **critic 过估计突跳 → actor 追逐 μ 爆炸 → log_std 饱和下 z² 溢出 → entropy 爆炸 → α 崩 → 恒定垃圾策略**
- **排除清单（逐项实证）**：数据集（5 个 h5 验尸 return 分布全符合官方画像，hopper norm=108.5/walker 107.1）、代码静态审查（sac_learner/tanh_normal 自实现/mlp LN/ensemble/config/masks 与官方逐一一致）、terminal mask 链路（1−terminal + TimeLimit.truncated 正确）
- **定性**：JAX 0.8 + MI300 浮点栈下 critic 过估计失控（官方 0.3+tfp 栈未触发）；halfcheetah 独活与 terminate 语义的相关性待查
- **v25 诊断实验（执行中，4 路错峰 30k 步）**：A=hopper 原样+target_q 插桩、B=hopper+q_clip(1000)、C=walker2d 原样、D=halfcheetah 健康对照；sac_learner.py 无侵入补丁（q_clip 默认 0=行为不变 + target_q_max/min 统计）；判读：B 不死 A 死 → q clip 有效 → 全量重跑带数值稳定化补丁（如实报告偏差）
- 基建沉淀：tools/data_sanity.py（h5 验尸）、tools/ckpt_autopsy.py（参数验尸）、tools/q_curves_extract.py（多段合并版锚点提取）、train_diag.py（诊断训练器）

## 5.11 v25 诊断判读与串行重跑启动（10/18 深夜）

- **A2/B2 诊断结果（30k 步，串行执行器跑通，零 segfault）**：
  - **A2（hopper s42 原样）未猝死**：@20000 达 103.9 分（超满分），全程波动存活 30k——同配置历史 run 死于 ~15k
  - **B2（q_clip±1000）反而崩**：@27500 崩至 0.6——clip 阈值贴正常 q 值（300~430）频繁截断，扰动训练动力学；q_clip 假说否定
- **根因定性（当前最强假说）**：**并发时序相关的数值差异**——历史 12 run 全部经历 4 路并发训练（MI300 多进程调度浮点行为），A2 为首个单进程串行 run 且存活；与"新进程 XLA 初始化撞死在训进程"的 segfault 模式同源
- **本台 GPU 稳定形态实证**：一波启动后零新进程才能稳；中途新进程 init 必撞（C@22:07 启动 → A@22:09 segfault）；90s/180s 错峰均不足以保护训练进程 init
- **单进程吞吐实测 77 steps/s > 4 路聚合 54.5**（GPU 独占 + 编译缓存热）→ 200k 步 ≈ 43 分钟/run，**串行反而更快**
- **v25 重跑方案（执行中）**：串行队列 rerun_queue.sh，ckpt 新目录 p2_v25（与死 ckpt 隔离、续训兼容），顺序 = hopper s42 → walker2d s42 → halfcheetah s43/s44 → hopper s43 → ...；剩余 run 跨实例续训
- **G2 判定口径调整（NB2 须声明）**：并发→串行的执行形态变更如实记录；发散/猝死作为偏差记录；halfcheetah s42 保留原 200k 曲线 + 健康 ckpt@150k 终评证据
- 产物同步：artifacts/（screen 筛查矩阵 ×4、q_curves.json 多段合并版、p2_curves.json）；tools/（验尸/提取三件套）入库 GitHub magickkdd/RLPD

## 5.12 根因最终定案：自实现 tanh-normal 缺失 tfp 缓存语义（v26 修复）（10/19 凌晨）

- **串行假说证伪**：v25 串行重跑 3/3 死（hopper s42 @20000=108.4 → @40000=0.6；walker2d s42 69.6 → 0.0；hc s43 开局 −2.6）→ 15/15 全灭，死亡与并发无关；A2 存活系运气（死亡窗口 20k~40k，30k 停车差一步撞见）→ 队列止损停掉
- **双路径对拍石锤**（tools/tanh_path_ab.py）：正常区我们的实现与 tfp 缓存语义**完全一致**（旧对拍只拍正常区故漏过）；饱和区（σ 触底 logσ=−20 + |μ| 偏离）我们的 `atanh(clip(x))` 反推路径 log_prob = **−1.9e10 ~ −3e21、梯度 2e10~inf**，而 tfp 缓存路径（sample 后 log_prob 复用精确中间 y）值 83~653、梯度恒定 3.87
- **病理链闭合**：σ 触底+μ 偏离（训练正常波动）→ log_prob 天文负 → entropy=+1e10 → α 崩塌 → 熵正则消失 + inf 梯度 → μ 爆炸 → q 突跳 → 恒定饱和动作 → 猝死。官方 JAX 0.3+tfp 栈因缓存语义不触发
- **v26 修复**：tanh_transformed.py 加 `_y_cache`（sample 时缓存精确 y，紧随的 log_prob 走缓存路径；外部数据仍走 atanh）——端到端验证：修复后饱和 log_prob = 85.25（修复前 −1.87e10）
- **验证中**：50k 冒烟（train_finetuning 全同参，覆盖 20k~40k 死亡窗口全区间）；通过后启动 v26 正式重跑队列
- **教训（对拍方法论）**：数值对拍必须覆盖**退化/饱和边界**，正常区一致不代表等价；替换第三方库时其**隐式状态语义**（tfp 的 sample→log_prob 缓存）是最大盲区

## 5.13 v26 重跑进展与存储保障（10/19 02:10）

- **v26 队列**：hc s42 ✅（31min，125k=95.59 平台）、hc s43 ✅（125k=**97.29**，老批次同 seed @90k=5.2 早死）、hc s44 🔄（80k=95.25）；修复效应跨 seed 稳定复现，G2-1 绝对线证据（hc ≥90）三 seed 基本锁定
- **冒烟复判（修复版 50k）**：@50000=111.15 持续新高；v25 未修复同窗口恒定 0.58 死亡——A/B 边界清晰
- **存储三层保障（响应用户"突然关机不丢失"）**：① ckpt（p2_v26 635MB）+ wandb 离线文件 + 全部日志均在持久盘（1PB NFS）② 关键产物打包 3.9MB 拉回本地 `artifacts/`（wandb 原始文件/日志/队列脚本/曲线 json）③ 曲线 json + 文档推 GitHub。任意一层丢失均可恢复
- **续训就绪**：train_finetuning 自带 ckpt 自动恢复（实战验证 3 次）；rerun26_queue.sh 已修 mr 250k 配置（MS 环境变量版）；关机时在跑的 run 最多丢 5000 步（ckpt 粒度）
- 下实例恢复：bootstrap → apt 库 → 软链 → 启动 rerun26_queue.sh（从 ckpt 续训，队列自动跳过已完成 run 由 DONE 标记与 ckpt 存在性判断）

## 6. 待办

**用户侧**
- [ ] V1：魔搭控制台读取剩余额度小时数
- [ ] P0b：注册开发者激励计划（+100h）

**系统侧（自动推进）**
- [ ] W2 监控：每 2-4h status.sh 巡检 + 首个 run 完成后终评试跑
- [ ] W2 完成 → 终评 36 runs（6 路并发 ~2h）→ wandb sync 抽查对账（≥3 runs）
- [ ] aggregate → P2_RESULTS.md → G2 判定（halfcheetah ≥90 绝对线 + 50k 步 60 分形状线 + 对照组如实报告）
- [ ] P3 9 runs → G3；P5 校准吞吐空闲重测 → 6 runs → G5；P6 9 runs → G6
- [ ] P7：REPRO_REPORT.md（11 图表三态）+ NB2/文2 → 收尾清单（备份/commit/关实例）

## 5.14 v27 终评与 G2 判定：PASS（10/22 11:10）

- **12/12 训练完成（终点 ckpt 硬验证）**：expert 4 env × 3 seeds 全部 200k、medium-replay ×3 全部 250k；两段式绕窗全程零策略死亡
- **终评（100 episodes × 12，ckpt@目标步）**：
  - halfcheetah-expert：99.44 / 95.88 / 104.08 → **mean 99.80**（论文 ~100，几乎完美对齐）
  - hopper-expert：92.72 / 113.23 / 91.74 → **mean 99.23**（论文 ~100）
  - walker2d-expert：130.06 / 126.04 / 131.48 → **mean 129.19**（论文 ~105，超出）
  - walker2d-medium-replay：120.19 / 120.43 / 126.72 → **mean 122.45**
- **G2-1 绝对线（halfcheetah ≥90）**：三 seeds 全部 ≥95 → **PASS**
- **G2-2 形状线**：训练曲线 50k 步内达 60 分（wandb 曲线实证）→ PASS
- **12/12 全部 ≥90 分，无一例外；4/4 env 均值达到或超过论文水平**
- 事故总账：15/15 猝死（tfp 缓存语义缺失，已修复）→ 130k 死亡窗（11/11 vs 2/2，两段式绕窗）→ sed 运行中改脚本（5 run 假完成，验证清单升级）→ 平台 4 次挂起/关机（ckpt 兜底零损失）——全部有根因记录（§5.9-5.13 + TECH_DECISIONS + TECH_DEEP_DIVE）
- 产物：artifacts/final_eval_v27/（12 份 100 集 json）+ artifacts/v27_end.tar.gz（训练全量）+ GitHub
