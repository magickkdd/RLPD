#!/usr/bin/env bash
# P0a 修复链: numpy2 折返 + mujoco-py 重编 + dmcgym(main) + 新 deployment
set -x
PIP=/mnt/data/miniforge3/envs/rlpd311/bin/pip
PY=/mnt/data/miniforge3/envs/rlpd311/bin/python

# 1) numpy 回 2.x（jax 0.9 需要 StringDType）
$PIP install -q 'numpy==2.2.6'
# 2) dmcgym main 分支
$PIP install -q https://codeload.github.com/ikostrikov/dmcgym/tar.gz/refs/heads/main
# 3) mujoco-py 按 numpy2 重编（apt 库已在位）
rm -rf /root/.cache/mujoco_py
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/mnt/workspace/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
$PY -c 'import mujoco_py; print("MP_NUMPY2_OK")'
$PY -c 'import jax, mujoco_py, gym, d4rl, dmcgym; print("ALL_IMPORTS_OK")'
echo FIX_DONE
