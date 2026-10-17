# P7 规格书：汇总对账 + 复现报告（G7）

> 版本: 2026-10-17 v2（v22 扫尾）· 上游: REPRODUCTION_PLAN.md v22 · 场地: 本机
> **⚠️ v24 范围裁剪（2026-10-18 02:30，用户决策，额度约束触发）**
> 输入清单收缩：P2_RESULTS 仅 RLPD 12 runs（36→12）、P3 仅 umaze 成功率、P5/P6 整节改"未复现（范围裁剪）"。
   三态表修订：Fig.4 行加"仅主方法、无对照"；Fig.19 行的 no-LN 面板改"未复现（范围裁剪）"；Fig.5/6、Fig.8/10/11/12 行改"未复现（范围裁剪）"。
   附录可加：3 个 SACfD halfcheetah seeds 白送对照曲线（完整数据，标注"范围外参考"）。报告口径差异声明增加"范围裁剪"一节。
> 判据影响：不依赖对照的主判据照常执行；依赖对照/消融的判据标注 **"未复现（范围裁剪）"** 入 NB2。
> 已完成资产保留：3 个 SACfD halfcheetah seeds（完整跑完，可作附录参考线）；续训体系见 RESUME.md §5.5。


## 1. 判据（G7）
- 每张复现图表标注 **"一致 / 偏差 / 未复现"三态结论**；偏差项有假设分析（checklist 已过）
- 交付 REPRO_REPORT.md + NB2/文2 提交材料（P0b +100h/50h 通道）

## 2. 输入清单（P7 开工归档到单一目录 /mnt/workspace/repro-report/ 后再写作）
| 输入 | 来源 |
|---|---|
| P2_RESULTS.md（36 runs 终评 + G2 判定） | P2 aggregate |
| P3_RESULTS.md（成功率 + CDQ 消融） | P3 aggregate |
| P4 冒烟记录 + 依赖链数据（无 aggregate 交付，弹性项除外） | P4 |
| P5_RESULTS.md（校准数据 + RLPD vs Online） | P5 aggregate |
| P6_RESULTS.md（LN / Seeded 消融） | P6 aggregate |
| P2_BENCHMARK.md（数据 sanity + 读图参考线） | W1-V5 |
| 训练曲线（.wandb / wandb sync 导出） | 各 run |
| 实例小时实际消耗 | 控制台读数（V1 起每次开机记录） |

## 3. 复现图表三态表（报告第 3 章的骨架）
| 论文图表 | claim | 覆盖阶段 | 初始态 | 备注 |
|---|---|---|---|---|
| **Fig.1/14（AntMaze teaser/完整结果）** | 1 | P3（umaze 成功率 + large-diverse CDQ 对照） | 待复现 | v21 修正：AntMaze 结果图 = Fig.14（teaser Fig.1）；原表 Fig.5 映射错误 |
| **Fig.2（LN 防 Q 发散，AntMaze Large Play + Pen Sparse）** | 3 | **未覆盖**（域不在矩阵） | **"未复现"**（v21） | claim 3 由 Fig.19 谱系部分覆盖；walker2d NaN 不能替代 |
| **Fig.4（主结果聚合 12 任务 ×250k）** | 1 | P2 子集（4 任务 locomotion） | 待复现 | **注明"4 任务子集、无聚合曲线"** |
| **Fig.5（V-D4RL vision-based）** | 6 | P5（cheetah-run medium 单任务子集；BC 基线未跑） | 待复现（子集） | v21 修正：Fig.5 是像素图归 P5（原表误标 AntMaze）；范围收缩如实声明 |
| **Fig.6（Pixels UTD 增益）** | 6 | 未安排 | **"未复现"** | |
| **Fig.7（LN 消融，Adroit/Expert-Adroit/Large-AntMaze/Humanoid）** | 3 | **未覆盖**（域不在矩阵） | **"未复现"**（v21） | claim 3 由 Fig.19 谱系部分覆盖 |
| **Fig.8（CDQ 消融，面板 = AntMaze Large Diverse）** | 5 | P3（large-diverse 域子集） | 待复现 | v21：消融域已对齐 large-diverse |
| Fig.9（Ensemble size 消融） | - | 未安排 | **"未复现"** | |
| **Fig.10/11（Seeded vs Symmetric，Pen Sparse / walker2d-medium）** | 2 | P6（walker2d-medium） | 待复现（部分） | Seeded(RLPD-config) 严格区别于 SACfD；pen 面板未跑 |
| Fig.12（offline_ratio） | 4 | 弹性项 | 视额度 | claim 4 标注"弹性" |
| **Fig.19（expert 消融，含 no LN）** | 3（谱系） | P2（RLPD/Online）+ P6（LN off） | 待复现 | v21 修正：Fig.19 属 claim 3 谱系（LN 消融），非 claim 1 |
| Table 1/2（超参） | - | 计划 §6 速查表 | **已对齐**（P2_SPEC F3 核查） | target_entropy 等逐项核对 |

## 4. REPRO_REPORT.md 结构（六章）
1. **执行摘要**：G2~G6 判据 PASS/FAIL 总览表 + 与论文差异一句话
2. **方法**：ROCm 适配摘要（指 NB1）+ 对照基线定义（RLPD/SACfD/Online/Seeded 四者严格区分）+ **口径差异声明**（v0/v2 数据替换与 best-in-dataset sanity、locomotion 100 eps 终评为扩展口径、3 seeds vs 论文 10 seeds、并发倍率实测过程）
3. **逐图表结果**：三态表逐行展开（曲线截图/表格）
4. **偏差分析**：每条 FAIL/偏差 → P2_SPEC §6.4 checklist 结论（实现正确性核验后仍偏差才定性）
5. **额度对账**：实例小时实际 vs **71~84 预算区间**（全方案 B 62 runs ≈ 113 run·h ÷ 倍率 1.6~1.35；v22 统一口径）+ W1 4-5h / W3 2h / 重跑余量；各阶段分解：P2 ~50 + P3 ~30 + P4 ~0.2 + P5 ~15 + P6 ~18
6. **复现指南**：环境重建（RESUME 链）+ runs.txt 生成 + xargs 跑批 + aggregate 全链——读者可照跑

## 5. 交付物
| # | 交付物 | 说明 |
|---|---|---|
| D7-1 | REPRO_REPORT.md | 上述六章；本机撰写 |
| D7-2 | NB2（Gallery Notebook） | **Gallery 环境无 rlpd311/ROCm/MI300X（核查 #30）→ cell 分层设计：数据/结果分析 cell（Gallery 可跑）+ 训练 cell（标注"需 AMD 实例，展示产物与命令"）** |
| D7-3 | 文2（开发者实践文章） | 复现方法论：倍率实测教训（2.53× 表观 → 1.33× 稳态）、误开 CPU 实例、数据文件名 basename 规则等真实过程 |
| D7-4 | 曲线图集 | 各 run evaluation/return 曲线 matplotlib 出图（aggregate 扩展 --plot） |

## 6. 风险
| 风险 | 对策 |
|---|---|
| 各阶段 aggregate 口径不一（return/成功率/归一化分） | aggregate.py 按 env 前缀自动切口径（antmaze=成功率、locomotion=get_normalized_score、adroit=REF 缺则原始 return） |
| 素材分散在实例/本地 | P7 开工先归档（§2 清单），实例侧文件 scp 回本地（project_memory 硬约束） |
| NB2 cell 非自包含 | 每个 cell 重跑验证（数据/ckpt 路径参数化） |

## 7. 收尾清单（P7 完成 = 项目收敛）
1. REPRO_REPORT.md 定稿 + NB2/文2 提交（回执进 RESUME）
2. 实例最终备份（ckpt/RESULTS/报告 → ModelScope 仓库 + 本地三层）
3. RESUME.md 终态更新；实例关闭（browser 或用户）
4. git 全量 commit（rlpd 仓库 train_baselines/final_eval 等新文件）
