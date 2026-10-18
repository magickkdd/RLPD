# RESUME — 实例重启恢复手册（2026-10-18 10:25 关机前终态 · 8h 实例）

## 关机时状态（持久盘 /mnt/workspace，重启不丢）
- 范围：RLPD-only 15 runs（P2 12 + P3 umaze 3）
- **P2 已完成 4/12**：halfcheetah s42/s43/s44 + hopper s42（200k 干净退出）
- 其余 8 个 run 全部有 ckpt（平均 ~168k/200k）：
  - hopper s43 @150k / s44 @195k
  - walker2d-expert s42 @195k / s43 @195k / s44 @125k
  - walker2d-medium-replay s42 @125k / s43 @125k / s44 @125k
- 剩余训练量 ≈ 440k 步 ≈ **2.5h**（4 路）

## 开机后步骤（网页终端，共 3 步）
1. `bash /mnt/workspace/.remote-access/bootstrap_ts.sh`（tailscale + sshd + 自动拉起守护）
2. `apt-get install -y -qq libgl1-mesa-dev libglew-dev libosmesa6-dev libegl1 patchelf`（系统库，每次开机必装）
3. `ln -sfn /mnt/workspace/.d4rl /root/.d4rl; mkdir -p /root/.mujoco && ln -sfn /mnt/workspace/mujoco210 /root/.mujoco/mujoco210`（软链）

守护自动：等 bootstrap 完成后 5 分钟内开波 → 断点续训剩余 8 run → 12/12 后写 W2_ALL_COMPLETE。

## 恢复后剩余工作
1. 等 12/12（~2.5h）
2. 终评：bash /mnt/workspace/w3_launch.sh（RLPD-only 12 个，已预部署）
3. aggregate --suite p2 → P2_RESULTS.md → G2 判定
4. P3 umaze ×3（runs_p3.txt 取 umaze 3 行，~4.6h）→ p3_final_eval
5. P7 报告（对比声明标"范围裁剪"；SACfD 3 seeds 白送对照入附录）

## 已知坑（详见项目记忆 Lessons Learned）
- 新实例：openssh-server 要装、authorized_key.pub 用直填真钥的 bootstrap（已修）
- 数据：~/.d4rl 软链断了会读空 → 若 find 不到 hdf5 先跑 download_data.sh 重下（~2min）
- 波次启动风暴：守护已是 v3 错峰版（90s），勿手动同启多 run
