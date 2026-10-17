# RESUME — 实例重启恢复手册（2026-10-18 02:35 关机前终态）

## 关机时状态（全部在持久盘 /mnt/workspace，重启不丢）
- 范围已裁剪：**RLPD-only 15 runs**（P2 12 + P3 umaze 3），对照/消融/Pixels/P4 全部砍掉
- P2 进度：3 个 SACfD 完整完成（范围外，白送保留）；RLPD 4 个在跑被中断，ckpt 断点：
  - halfcheetah s42 @130k / s43 @130k / s44 @120k / hopper s42 @125k（均 200k 目标）
  - hopper s43/s44、walker2d-expert ×3、walker2d-medium-replay ×3 未开始
- P3 umaze ×3 未开始

## 开机后你要做的（只有一步）
```
网页终端: bash /mnt/workspace/.remote-access/bootstrap_ts.sh
```
最后会自动拉起主控守护 w2_master.sh → 自动重建未完成清单（runs_p2.txt = 12 行 RLPD）→ 4 路开波 → 崩溃/中断的 run 从 ckpt 自动续训 → 全部完成后写 W2_ALL_COMPLETE 到 logs/w2_master.log。

## 恢复后剩余工作（agent 侧接手）
1. W2 剩余 ~11 个 run（~10h）
2. 终评：gen_runs --suite p2_final_eval 36 行中**只跑 grep "train_finetuning.py" 的 12 行**（RLPD-only）
3. P3 umaze 3 runs（runs_p3.txt 9 行中取 umaze 3 行，~4.6h）→ p3_final_eval umaze 3 行
4. aggregate（--suite p2 / p3）→ P2_RESULTS.md → G2 判定（绝对线 + 形状线；对比声明标"范围裁剪"）
5. P7 报告 + NB2（记录：范围裁剪、v0/v2 数据口径、在线 buffer 不可恢复、SACfD 3 seeds 白送对照）

## 本地已同步
- instance_logs/（w2_master.log、heartbeat.log 快照）
- 代码/文档 GitHub: magickkdd/rlpd-repro（master）+ magickkdd/rlpd-rocm（rocm-adapt）
