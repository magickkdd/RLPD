# RESUME — 实例重启恢复手册（2026-10-19 03:42 关机前终态 · v26 修复版重跑中）

## 关机时状态（持久盘 /mnt/workspace，重启不丢）
- **代码已修复（v26 核心）**：`rlpd/rlpd/distributions/tanh_transformed.py` 已加 tfp 缓存语义（`_y_cache`）——15/15 猝死根因的修复，**不可回退到旧版**（备份在 /mnt/workspace/tanh_transformed.py.bak_v25）
- **v26 重跑（ckpts 在 /mnt/workspace/ckpt/p2_v26/RLPD/）**，队列日志 /mnt/workspace/rerun26.log：
  - halfcheetah s42 ✅ 200k（125k=95.59 平台）/ s43 ✅ 200k（97.29）/ s44 ✅（~02:35 完成，80k 已 95.25）
  - hopper s42 ✅（~03:07 完成）
  - walker2d s42：关机时 ~65k+ 步带 ckpt（03:07 起跑）
  - **未开始**：hopper s43/s44、walker2d s43/s44、walker2d-medium-replay s42/s43/s44（**max_steps=250000**，队列脚本已修）
- 旧批次的 12 run（死策略）在 ckpt/p2/RLPD/，仅作证据，勿用于终评
- 训练日志：/mnt/workspace/logs/v26_*.log；wandb 离线：/mnt/workspace/rlpd/wandb/（本地+GitHub 有备份）

## 开机后步骤（网页终端，共 3 步）
1. `bash /mnt/workspace/.remote-access/bootstrap_ts.sh`（tailscale + sshd；若实例重建过，IP 可能变化需同步本地 SSH 命令）
2. `apt-get install -y -qq libgl1-mesa-dev libglew-dev libosmesa6-dev libegl1 patchelf`（每次开机必装）
3. `ln -sfn /mnt/workspace/.d4rl /root/.d4rl; mkdir -p /root/.mujoco && ln -sfn /mnt/workspace/mujoco210 /root/.mujoco/mujoco210`

## 恢复队列（SSH 或网页终端执行，串行单进程，勿并发）
```bash
# 队列脚本已带 ckpt 自动续训（train_finetuning 检测最新 ckpt 接续）；
# mr 的 250k 用 MS 环境变量；已完成 run 由 ckpt 存在性自然跳过续跑开销（从 200k ckpt 恢复后立即到步退出）
# 注意：脚本按顺序全量重列，已完成 run 会"restore→即刻退出"秒过
nohup bash /mnt/workspace/rerun26_queue.sh > /mnt/workspace/logs/rerun26_resume.log 2>&1 &
```
- 已完成 run（hc 42/43/44、hopper 42）会秒级跳过；从 walker2d s42 的 ckpt 接续
- 速率 ~70 steps/s，每 200k ≈ 31 分钟；剩余 7 run + mr 250k×3 ≈ 4.5h

## 恢复后剩余工作
1. 等 v26 12/12（~4.5h）
2. 终评：final_eval.py（**用 --ckpt_step 或默认最新 ckpt**；注意 TF 先行 import 已内置）100 集 ×12
3. aggregate → P2_RESULTS.md → G2 判定（hc ≥90 三 seed 已实证 95+，预期 PASS）
4. P3 umaze ×3（spec 未变，代码已修复版）→ G3
5. P7 报告：NB2 必须写明 v26 修复（tfp 缓存语义）与并发批次猝死的完整偏差链（STATUS §5.10-5.13）

## 已知坑（详见项目记忆 Lessons Learned）
- 新实例：openssh-server 要装；数据软链断了跑 find 验证 + download_data.sh
- **勿并发**：本机 GPU 初始化撞车（一波启动后零新进程）；训练串行最快（70 vs 4路聚合 54.5 steps/s）
- 终评前确认 `import tensorflow` 在 flax.checkpoints 之前（final_eval.py 已内置）
- wandb 曲线提取用 tools/q_curves_extract.py（注意按 env 匹配 REF，勿跨 env 用系数）
