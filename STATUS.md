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
