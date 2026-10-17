#!/bin/bash
# P4: mj_envs 装机 v3（clone 放 site-packages 之外, 避免命名空间包遮蔽）
set -e
SP=/mnt/data/miniforge3/envs/rlpd311/lib/python3.11/site-packages
PY=/mnt/data/miniforge3/envs/rlpd311/bin
rm -rf $SP/mj_envs $SP/mj_envs.egg-link $SP/__editable__*mj_envs* /mnt/workspace/mj_envs_repo
$PY/pip uninstall -y mj_envs -q 2>&1 | tail -1 || true
grep -v mj_envs $SP/easy-install.pth > /tmp/eip 2>/dev/null && mv /tmp/eip $SP/easy-install.pth || true
mkdir -p /mnt/workspace/mj_envs_repo
tar xzf /mnt/workspace/mj_envs_full.tar.gz -C /mnt/workspace/mj_envs_repo
cd /mnt/workspace/mj_envs_repo/mj_envs
$PY/pip install -e . --no-deps -q 2>&1 | tail -2
cd /mnt/workspace/rlpd
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/mnt/workspace/mujoco210/bin:$LD_LIBRARY_PATH
$PY/python -c "
import gym, mj_envs
env = gym.make('pen-binary-v0')
env.reset()
print('PEN_BINARY_OK', env.spec.max_episode_steps)
"
