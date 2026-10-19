# TECH_DECISIONS — 技术选型决策记录（RLPD 复现项目）

> 2026-10-19 · 所有决策均有项目实测数据支撑，可追溯至 STATUS.md §1-5.13 与项目记忆 Lessons Learned

## 1. 算法基座：官方 RLPD 仓库 fork 适配（ikostrikov/rlpd）

**选定**：fork 官方 JAX 实现，仅做 ROCm/JAX 0.8 API 适配（~10 处），算法逻辑零改动。

**核心原因**：复现研究的金标准是"与论文作者代码逐行对齐"。官方 repo 是论文的 ground truth 实现（10-Q ensemble、LayerNorm critic、UTD=20 对称采样、SAC 骨干），fork 它等于把"复现偏差"的上界压缩到"适配差异"，排障时可自信地排除算法实现嫌疑——本次排障正是因此才能快速锁定根因在分布库适配层（§3）而非算法本体。

**替代方案对比**：
- 从零自实现：完全掌控但引入实现误差风险；RLPD 细节多（REDQ min、backup_entropy、pretrain 段），从零写极易引入隐性偏差，且复现声明从"官方代码适配"降级为"自研实现"。未选：偏差无法归因。
- CleanRL SAC 扩展：单文件易读，但无 ensemble/对称采样现成实现，改造量≈重写。未选：失去官方对拍锚点。
- DeepMind Acme：工程质量高但 TF 深度耦合、无现成 RLPD agent、依赖重。未选：与 JAX 轻量诉求冲突。

**局限与预案**：官方代码假设 NVIDIA + JAX 0.3，API 断裂需逐个适配（已发生，如 FrozenDict.copy 移除）。预案：适配点全部注释标记 rocm-adapt，保留 upstream 对照分支（GitHub rlpd-rocm 的 main vs rocm-adapt）。

## 2. 深度学习框架：JAX 0.8.x on ROCm 7.2.3（放弃 JAX 0.3）

**选定**：JAX 0.8.x + ROCm 7.2.3 wheels，AMD MI300X（192GB）。

**核心原因**：平台硬约束——魔搭免费实例只提供 MI300X + ROCm 7.x，JAX 0.3.x（官方 pin）无 ROCm 7 wheel。实测 0.3→0.8 API 断裂仅 ~10 处且全部可适配，算法数值行为经对拍验证（pretrain 段 loss 曲线 1e-6 级对拍通过）。MI300X 192GB 对单进程 UTD=20 训练（9.5GB RSS / VRAM 76%）严重过剩，为长队列与失败重试提供余量。

**替代方案对比**：
- PyTorch 重写：ROCm 成熟、生态最大，但整个算法重写（数千行）、全部官方对拍锚点失效，复现声明降级为"自研实现"。未选：与复现目标根本冲突。
- TensorFlow 线：生态萎缩，JAX 版 RLPD 是社区主流。未选：无增益。
- CPU 训练：无兼容问题但 UTD=20 算力需求下不可接受（GPU 实测 66-77 it/s）。未选：时间成本爆炸。

**局限与预案**：JAX 0.8 + ROCm 组合存在未文档化数值差异（130k 死亡窗疑与此相关）。预案：两段式绕窗（§6）+ 全量 wandb 曲线留证 + 报告如实声明栈差异。

## 3. 概率分布库：~70 行纯 JAX 自实现 tanh-normal（弃用 tensorflow-probability）

**选定**：rlpd/distributions/tanh_transformed.py 自实现 MultivariateNormalDiag + TanhTransformedDistribution（sample/log_prob/mode 三接口）。

**核心原因**：tfp 与 JAX 0.8 无兼容版本，而 actor 梯度完全依赖 tanh-normal 的 reparameterized sample 与 log_prob。自实现体量小、数学形式标准（log det 用数值稳定形式 2(log2 − y − softplus(−2y))）、可与官方行为逐项对拍。**本项目最大教训在此**：初版对拍只覆盖正常区（通过），漏掉饱和区（σ 触底+μ 偏离）下 tfp 的隐式缓存语义（sample 后 log_prob 复用中间量 y）——缺失导致 log_prob 爆到 1e21、梯度 inf，引发 15/15 训练猝死。修复 = 复刻缓存语义（_y_cache），修复后饱和区 log_prob 85.25（修复前 −1.87e10）。

**替代方案对比**：
- tfp-nightly：官方行为零偏差，但与 JAX 0.8 兼容无保证，且引入整个 TF 依赖树（实例已证实 TF 混装触发 orbax 二次初始化 segfault）。未选：依赖地狱风险 > 收益。
- PyTorch distributions 局部混用：行为成熟，但跨框架张量转换破坏纯 JAX jit 链。未选：架构污染。
- 手写但不含缓存语义（初版）：代码最简但饱和区爆炸。已被证伪。

**局限与预案**：隐式语义对拍依赖"想得到要拍什么"。预案：对拍工具固化（tools/tanh_path_ab.py），覆盖正常/临界/饱和三区 × 值/梯度两维；任何第三方库替换必须先做退化边界对拍（已入项目记忆硬规则）。

## 4. 算力平台：魔搭社区免费 MI300X Notebook

**选定**：魔搭免费 AMD MI300X 实例（192GB VRAM / 200GB RAM / 23 核），多次开关实例接力完成全部训练。

**核心原因**：零成本 + 顶级单卡（单进程 76 it/s，200k 步 ~40 分钟）+ 1PB NFS 持久盘允许断点接力。总训练量（含全部重跑）>20h 完全被免费额度覆盖，且"随时可能关机"倒逼出断点续训体系（§6）——该体系在 15+ 次死亡/停机中兑现价值。

**替代方案对比**：
- Colab/Kaggle：零成本但 T4/V100 16GB 级算力，且 Colab 无 AMD/ROCm、Kaggle 会话受限。未选：算力差一个量级且跑不了 ROCm 栈。
- AutoDL 租卡（4090/A100）：稳定快速，但本项目预算为零（用户明确免费额度优先）。未选：预算硬约束。
- 本机：无 MI300X 级显卡。不可行。

**局限与预案**：实例随机停机/挂起（已遇 3 次）、单实例 8h 额度、网页终端偶发冻结。预案：ckpt 每 5k 自动存 + resume（丢步上界 5k）；三层备份（§10）；bootstrap 自动化使重启恢复成本 <5 分钟。

## 5. 数据集获取：curl 直抓 hf-mirror resolve URL

**选定**：download_data.sh 用 curl 按 URL 清单直抓 hf-mirror 的 hdf5，落持久盘 + ~/.d4rl 软链，自带断言与重下。

**核心原因**：国内实例出网受限——d4rl 官方源（rail.eecs.berkeley.edu）不可达；huggingface_hub 2.x 的 hf download 无视 HF_ENDPOINT 直连官方站（实测超时）。curl+镜像是最小依赖路径，且可 h5 元数据校验完整性（tools/data_sanity.py 轨迹 return vs 官方画像：hopper-expert norm=108.5、walker2d 107.1）。

**替代方案对比**：
- d4rl 库自动下载：零配置但官方 URL 硬阻断。未选。
- hf CLI + HF_ENDPOINT：官方姿势但 2.x 实测无视该变量，且 HF search API 分页截断+子串误匹配。未选：不可靠。
- 本机下载再 scp：同样境外不稳且 1GB 级慢。未选。

**局限与预案**：镜像文件命名与 d4rl 期望名不同（无 -v0 后缀），版本错配风险曾列头号嫌疑（后经数据验尸排除）。预案：data_sanity.py 纳入开机冒烟清单，新数据集先验尸再训练。

## 6. 训练可靠性：orbax ckpt（keep=20，每 5k 步）+ resume 补丁 + 两段式绕窗

**选定**：train_finetuning.py 启动时检测 {log_dir}/checkpoints 最新 ckpt 自动 restore 续训；v27 起每个 run 拆两段：0→125k 停车，再从 ckpt 续训到目标（200k/250k）。

**核心原因**：两层。(a) 实例随时停机 → 断点续训是生存必需（丢步上界 5k，实战兑现 15+ 次）。(b) 本项目发现**从 0 起跑的训练在第 126~131k 步确定性 segfault（11/11，跨 4 env 跨 seed），而从 @125k ckpt 续训稳定存活（2/2，等效步数完全绕开窗口）**——触发机制未明（墙钟 ~31 分钟与步数 ~130k 两个影子共存），但经验规律足以支撑两段式绕窗工程对策：第一段窗口前停车，第二段等效步数全部穿窗。这是把"未解之谜"转化为"可执行对策"的典型案例。

**替代方案对比**：
- 整跑重跑：流程最简，但与实例停机现实冲突，且 130k 窗口下 11/11 全灭等于永远无法完成。未选：不可行。
- wandb/自研 pickle ckpt：灵活，但 orbax 是 flax 官方路径，restore 语义（target=agent 结构还原）经叶级对拍 36/36 零差异验证。未选：无增益引风险。
- 等待根因定位再跑：学术上最干净，但触发机制在 JAX/ROCm 栈内排查成本不可估。未选：经验对策先行，根因研究入 backlog。

**局限与预案**：两段式边界（125k）是经验值，换实例/环境后窗口可能漂移需重标定。预案：队列内建资源监控（q_monitor：RSS/存活时长），死亡即有现场；ckpt resume 幂等，重跑队列自动跳过完成段。

## 7. 实验编排：串行队列脚本（弃用 4 路并发）

**选定**：单进程串行队列（rerun27_queue.sh 逐 run 执行）+ 独立资源监控循环 + faulthandler 全程开启。

**核心原因**：三组实测数据。(a) 吞吐：单进程 76 it/s > 4 路聚合 54.5 it/s（本台 GPU 调度开销），串行反而快。(b) 稳定：多进程同时 XLA 初始化触发 runlist 过订阅 segfault（4 路全灭 × 2 波实测），90s/180s 错峰均无法保护训练进程 init。(c) 排障：串行让每次死亡的归因空间减半（无并发干扰变量）。

**替代方案对比**：
- 4 路并发（V4 曾裁定 -P 4）：理论聚合吞吐略高（实测反而略低）。未选：稳定性完败且速度无优势。
- 主控守护自动开波（w2_master）：崩溃自愈，但多代孤儿进程互踩事故（同 seed 双写），错峰窗口仍是雷区。未选：简单队列 + 幂等 ckpt 检查已覆盖其价值。
- Slurm/Airflow：单机单卡场景过度工程。未选。

**局限与预案**：串行拉长总墙钟（12 run ≈ 7h）。预案：两段式 + ckpt 幂等使中断可无缝接续，总墙钟可跨实例摊薄。

## 8. 实验追踪：wandb offline + 锚点字节扫描解析

**选定**：训练全程 WANDB_MODE=offline，事后 tools/q_curves_extract.py 锚点扫描提取曲线（q/temperature/entropy），deployment/aggregate.py 出报告。

**核心原因**：离线模式不受实例网络波动影响；wandb protobuf 记录训练内评估、q 值、熵等全部指标——**本次排障的决定性证据（q=308→1007 突跳、entropy 爆到 8e24）全部来自它**。wandb 0.30 移除了 Python datastore API，锚点扫描（字节模式 + _step 偏移配对）是逆向出的最小解析方案，已固化为多段合并版工具。

**替代方案对比**：
- TensorBoard：格式开放解析容易，但需改训练代码换记录层、丢 wandb 生态。未选：侵入式改造不值。
- 纯 CSV 自记：零依赖但丢 run 管理/配置快照。未选：无增益。
- wandb 在线模式：云端实时，但国内实例到 wandb 服务器不稳。未选：网络不可靠。

**局限与预案**：锚点扫描对 wandb 版本敏感（0.30 格式）；同 step 双份记录（history+summary）需配对容错。预案：解析工具带 env 检测与 REF 匹配（曾修复跨 env 用错归一化系数 bug），输出带 n 点数校验。

## 9. 远程通道：Tailscale userspace networking（自建 socket + serve 转发）

**选定**：实例内 bootstrap_ts.sh 自动注册 tailscale 节点 + serve --tcp=2222→2223，本机经自定义 socket ProxyCommand SSH 直连。

**核心原因**：魔搭无公网 SSH 入口，网页终端偶发冻结不可作为操作通道。Tailscale userspace 模式（无需 root 网络权限、自建 socket 避免与系统态冲突）在魔搭容器内稳定工作；serve 把 sshd 藏在 2223 高位。实战 3 个实例 × 多天，除实例自身停机外零故障。

**替代方案对比**：
- 魔搭网页终端：零配置但偶发冻结（官方已知）、不可编程。未选：只作应急。
- frp/自建反代：可控但需公网 VPS 中转（成本+暴露面）。未选：无免费可靠 VPS。
- ZeroTier：同类产品，国内容器 NAT 穿透与 relay 质量实测不如 tailscale DERP(hkg)。未选。

**局限与预案**：实例重建后节点需重注册（bootstrap 已自动化）；本地 WSL 存在同名假节点（回环陷阱）→ 巡检已固化身份校验（/mnt/workspace 存在性 + hostname 比对）。

## 10. 产物安全：三层备份（持久盘 → 本地 → GitHub）+ git bundle 中转

**选定**：代码与文档实时 git（本地 commit → bundle → scp 到实例 → 实例出网 push GitHub）；训练产物定期打包 scp 回本地 artifacts/；ckpt 与 wandb 原始文件以持久盘为一级存档。

**核心原因**：实例随时消失是常态而非异常（3 次停机实证）。三层独立失效模式：持久盘随实例销毁、本机网络抖动、GitHub 单点——三层齐备后"任何一层丢失均可恢复"（实测：实例两次停机期间，本地与 GitHub 产物支撑了全部分析继续）。git bundle 中转解决本机直连 GitHub 超时（443 连接 135s 不通）而实例出网正常的不对称。

**替代方案对比**：
- 仅实例持久盘：零操作但实例销毁即全失。未选：单点。
- 本机直推 GitHub：流程简但 TLS 路径不稳（实测超时）。未选：中转实测更可靠。
- 全量 ckpt 推 GitHub：最强冗余但 GB 级二进制 + 单文件 100MB 限制。未选：收益不成比例。

**局限与预案**：GitHub PAT 一次性使用（不入 config/文件）；bundle 中转已脚本化，成本 <2 分钟。

## 11. 评估口径：d4rl normalized ×100 + 100 episodes 终评 + 硬验证清单

**选定**：归一化分数按 d4rl v0 REF 表（get_normalized_score 0~1 → ×100 对齐论文图 19）；终评每 run 100 episodes（论文 B.1 的 100-trial 规则本只规定 AntMaze/Adroit，locomotion 系为扩展口径，NB2 声明）；**完成判据 = checkpoint_{max_steps} 存在 + 曲线尾部到步 + tqdm 分母核对**（DONE_0 不算完成）。

**核心原因**：口径错一档结论全废。v0 REF 与 v2 数据存在已知偏差（halfcheetah best_norm=92.9 <95，V5 已记录）；100 集对齐论文统计强度；硬验证清单直接来自"DONE_0 假完成"事故（tqdm 分母 180001/75001 的发现路径）。

**替代方案对比**：
- 原始 return 报告：免换算但跨任务不可比。未选。
- 10 episodes 终评：快但 ±10 分噪声淹没结论。未选：论文对齐需 100 集。
- v2 数据/REF 口径：更现代但与论文（v0）不可比。未选：复现以论文口径为准。

**局限与预案**：v0 口径已知偏差（数据 best_norm 与 REF 不一致）在报告声明；死亡窗相关 run 一律以曲线+ckpt 双证据标注有效训练步数。

## 12. 环境管理：conda 持久化 + apt 每次重装 + 冒烟门

**选定**：conda env 与 mujoco210/cymj 编译缓存放持久盘（免重装）；apt 系统库（libgl/glew/osmesa/egl/patchelf）每次开机重装（~1 分钟，脚本化）；任何队列启动前强制 import 冒烟。

**核心原因**：区分"贵"与"便宜"的重装——conda 依赖解析+cymj 编译是小时级（持久化），apt 二进制包分钟级（无成本）。冒烟门（import tensorflow; from rlpd.agents import SACLearner）直接来自三次"脚本上队列后秒死"事故，成本 10 秒，拦截小时级浪费。

**局限与预案**：持久盘 conda 与实例 glibc 强耦合（换基础镜像需重建 env）。预案：requirements.txt + 环境重建脚本入 deployment/，最坏 40 分钟重建。

---

## 附：决策方法论沉淀（本项目最高价值产出）

1. **数值对拍必须覆盖退化/饱和边界**——正常区一致 ≠ 等价；tfp 的隐式状态语义（sample→log_prob 缓存）是自实现替代的最大盲区。
2. **运行中修改正在执行的 shell 脚本 = 未定义行为**（bash 按字节偏移逐行读取）——5 个 run 假完成的根因。
3. **完成判据必须硬核**：DONE_0、平台分数都不是完成；ckpt 终点存在性 + tqdm 分母核对 + 曲线尾部三证合一。
4. **"未解之谜"可转化为"经验对策"**：130k 死亡窗机制未明，但 11/11 vs 2/2 的规律足以支撑两段式绕窗让项目继续——务实主义与根因研究并行。
5. **单进程串行在本场景全面优于并发**：吞吐更高、归因更干净、故障域更小。
