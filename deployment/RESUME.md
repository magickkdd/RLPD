# 恢复手册（实例重启/重开后必读）

> 更新: 2026-10-17 v17 · 状态: P0a(G0a ✅) / P1(G1 ✅ 带倍率复测尾巴)；**P2 规格 v2 已定稿（P2_SPEC.md），W1 开工即执行 V1~V5 验证**。**实例已于 2026-10-17 ~11:45 手动关闭**（收尾检查通过：无残留进程、资产完整），下次开机按 CONNECT.md 重连即可

## 实例当前状态（存于 /mnt/data，与 /mnt/workspace 同一 NFS 持久盘）

| 资产 | 路径 | 状态 |
|---|---|---|
| miniforge3 + rlpd311 环境 (py3.11) | /mnt/data/miniforge3/envs/rlpd311 | ✅ jax 0.9.2 ROCm 四件套 + flax 0.10.6 + optax 0.2.4 + numpy 2.2.6 + mujoco-py(numpy2 重编) + d4rl 1.1 + mjrl(aravindr93) + dmcgym(main, --no-deps) |
| mujoco210 | /mnt/workspace/mujoco210 | ✅ |
| 数据集目录 | /root/.d4rl → /mnt/workspace/.d4rl | ✅ halfcheetah_expert.hdf5（=halfcheetah-expert-v0.h5, 237MB v2 数据, md5=34cc91a3…7d）已在位 |
| 代码 + 部署脚本 | /mnt/workspace/{rlpd,deployment} | ✅ 最新 |
| SSH/tailnet 身份 | /mnt/workspace/.remote-access/ | ✅ tailscaled.state 持久，IP=100.78.127.38（CONNECT.md） |

**P0a 冒烟最终 9 PASS + 1 SKIP**（2026-10-17）：1-jax-gpu ✓ 2-mlp-jit ✓ 3-frozendict ✓ 4-tanh-normal ✓ 5-mujoco-py ✓ 6-rlpd-agent(agent.update 真实跑通, critic_loss≈48-75) ✓ 7-d4rl-dataset(GET_DATASET_OK (999000, 17)) ✓ 9-persist ✓ 10-train-import ✓；8-modelscope=SKIP（可选件，不影响训练）。

**G0a 已通过**（2026-10-17）：`train_finetuning.py --env_name=halfcheetah-expert-v0 --max_steps=1000 --start_training=0 --seed=42` 全链路实跑（wandb offline run-821880ye，_runtime≈28s）。指标：critic_loss 51.46→2.52、q 0.52→43.0、temperature 1.0→0.749、entropy -2.3→3.68、evaluation/return=196.65、training/return=-243.36（均为健康 SAC 早期特征）。**判读要点：默认 `start_training=10000`，`max_steps=1000` 时整程只随机采动作 + i=0 一次 eval，约 16s 正常退出且 stdout 无输出——这是正常行为不是崩溃**；`--start_training=0` 才能覆盖 agent.update 在线路径。wandb 离线指标提取：.wandb 为 protobuf，HistoryItem 结构 `\x12<len>key \x82\x01<len>value_json`。

**关键版本链（血泪教训）**：py3.11 + jax 0.9.2 + **numpy==2.2.6**（jax 0.9 要 StringDType；mujoco-py 按 numpy2 重编——换 numpy 后必须删 `site-packages/mujoco_py/generated/cymj*_311*.so` 与 `_pyxbld*` 再 import）+ flax==0.10.6（0.9.0 太老）+ optax==0.2.4 + d4rl==1.1 + **mjrl(aravindr93/mjrl，--no-deps)** + dmcgym(main 分支，**--no-deps --no-build-isolation**)。

**新容器重建后 apt 重装清单**：`apt-get install -y libgl1-mesa-dev libglew-dev libosmesa6-dev libegl1 patchelf openssh-server`；符号链接 `ln -sfn /mnt/workspace/mujoco210 /root/.mujoco/mujoco210`；`git config --global http.version HTTP/1.1`。

## 重启后恢复步骤

1. **连接实例**：完全按 [CONNECT.md](CONNECT.md) 三步走（WebIDE 终端 → bootstrap 一行 → 本机 connect_notebook.sh）
2. 若持久盘完好（默认），环境直接可用；如需跑训练，source 环境：
   ```bash
   export PATH=/mnt/data/miniforge3/envs/rlpd311/bin:$PATH
   export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/mnt/workspace/mujoco210/bin
   ```
3. 若持久盘被清空（罕见）：ModelScope 私有仓库拉 bundle → apt 清单（上）→ `pip install -r deployment/requirements-rocm.txt` → mjrl/dmcgym 特殊装法（本文件下方『已知坑』）→ mujoco-py numpy2 重编检测（fix_numpy2.sh 逻辑）。**install_step1/2.sh、setup_amd.sh 已 DEPRECATED 勿执行**

## 下次会话待办（按序）

1. ~~重跑冒烟~~ ✅ 9 PASS + 1 SKIP；~~G0a 收尾~~ ✅ 已通过（见上）
2. ~~P1 吞吐校准~~ ✅ **已完成（2026-10-17）**：
   - UTD=1（sac 默认 config）基线：5000 步 / 29s，稳态 **~2ms/env-step ≈ 500 steps/s**
   - **UTD=20 + 论文 config（rlpd_config: E=10, M=2, LN=true）实测：2000 步 / _runtime=128s → 稳态 ~33ms/env-step ≈ 30 steps/s；单梯度步 ~1.7ms 与 UTD1 基线一致（吞吐线性于 UTD）**
   - 外推：250k 步主 run ≈ **2.4h**；UTD20 下 q -1.6→692（学习快），指标健康
   - 并发曲线（2k 步）：1/2/4/6 路 = 168/193/334/398s，表观吞吐 2.53×@6 路——**v17 修正：含 ~50% 非稳态开销，稳态 ≈1.33×，真值待 W1-V4（20k 步 + rocm-smi）重测**
   - 预算表 §4 与方案 B 额度结论已按实测修订（REPRODUCTION_PLAN.md v16）
3. **下次开机从 P2 W1 开始（规格：P2_SPEC.md v2 §4-§5）**，顺序：
   - **先跑 bash deployment/upload_all.sh（一键同步全部交付物+语法检查）**
   - V1 读控制台实际剩余额度写入本文件；V2 ckpt save→restore 往返；V3 nohup >1.5h 无人操作存活（被关 → agent Schedule 每 45min 心跳保活）；V4 倍率重测（20k 步 × 单/4/6 路 + rocm-smi）；V5 数据 best-in-dataset sanity
   - 交付物已就绪（本地 v21）：train_baselines.py（sacfd=plain SAC+seeded+UTD1 固定容量 buffer / online=rlpd_config 无离线）、final_eval.py（restore_checkpoint，json 写 exp_prefix 层，normalized ×100）、gen_runs.py（**--suite p2/p3/p4/p5/p6 全阶段**）、aggregate.py v3（REF 表归一化判据）、status.sh、download_data.sh（含 ~/.d4rl symlink 断言 + walker_mixed.hdf5 等正确文件名）
   - 数据：hopper_expert / walker2d_expert / walker_mixed / walker2d_medium / antmaze v2 同名文件（download_data.sh 一键）
   - W2 主批：`python deployment/gen_runs.py --suite p2 --out runs_p2.txt && xargs -P 6 -d '\n' -I CMD bash -c CMD < runs_p2.txt`（36 runs）；SACfD=plain SAC+seeded+**UTD1**（v21）
   - 归一化约定：d4rl 返回 0~1，**入库/对账一律 ×100**（0~100 论文口径）
4. 注意：实例 **1 小时无操作自动关机**；长训用 `setsid nohup ... &` 脱离会话 + 日志轮询；wandb offline 指标可离线解析 .wandb（方法见上）或 `wandb sync`

## 已知坑（实战验证汇总）

- **numpy 版本拉扯**：jax 0.9 要 numpy≥2，mujoco-py 老教程说 numpy<2——正解 = numpy 2.2.6 + mujoco-py 重编（.so 在 site-packages/mujoco_py/generated/，不在 ~/.cache）
- **flax 版本**：0.9.0 配 jax 0.9 挂（EvalTrace/level、partial.init）；0.10.6 OK；partial 不能直接 .init，须先调用成实例（rlpd agent 写法 `TanhNormal(...).init()`）
- **d4rl 1.1 隐藏依赖 mjrl**（报错被 `d4rl/__init__` try 吞成 warning，注册表只剩 bullet 系）；正确仓库 aravindr93/mjrl；git clone 不稳 → 本地下 tarball scp
- **dmcgym**：master 404 用 main；pip 装它会重建 mujoco-py 失败 → `--no-deps --no-build-isolation`
- jaxlib 0.8.x/0.3.x 已从官方源删除；JAX ROCm = PyPI 四件套同版本 pin
- conda activate 在 nohup 非 shell 不可靠 → 绝对路径；pip 守卫先核对目标 env
- GitHub clone `curl 16 HTTP2 framing` → `git config --global http.version HTTP/1.1`；clone 抖动 → 本地下 tarball scp
- mujoco_py 要求 `~/.mujoco/mujoco210/bin` 在 LD_LIBRARY_PATH（symlink 后仍需显式 export）
- WebIDE 终端偶发冻结 → 刷新重开；现在有 SSH，优先走 SSH；computer use 仅用于开关实例
- **mujoco210 实际路径 = /mnt/workspace/mujoco210**（env.sh v17 已修正；/mnt/data 与 /mnt/workspace 同一 NFS 但脚本勿混用）
- i=0 必触发一次 eval（i%eval_interval==0 无法用 flag 关闭）——2k 步短基准中占比极大，测吞吐须 ≥20k 步
- **开机必须选 192G 显存的 AMD GPU 实例**（2026-10-17 曾误开 CPU 实例：无 /opt/rocm、/dev/kfd，jax 仅 CpuDevice；数据在 NAS 不丢，但白耗一次开机周期）
