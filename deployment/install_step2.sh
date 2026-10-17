# DEPRECATED (v17): 本脚本含过期配置（py3.12/flax0.9/tensorflow-cpu/dmcgym master/jax0.8 错误路径），勿执行。
# 恢复环境一律按 RESUME.md『恢复步骤』：requirements-rocm.txt + apt 清单 + mjrl/dmcgym 特殊装法。
#!/usr/bin/env bash
# pip step 2: py3.11 完整训练环境 (jax 0.9.2 ROCm + d4rl 老栈)
set -x
cd /mnt/workspace
/mnt/data/miniforge3/bin/conda create -y -p /mnt/data/miniforge3/envs/rlpd311 python=3.11
PIP=/mnt/data/miniforge3/envs/rlpd311/bin/pip
PY=/mnt/data/miniforge3/envs/rlpd311/bin/python
$PIP install 'setuptools<66' 'cython<3' 'numpy<2' wheel
$PIP install gym==0.23.1
$PIP install mujoco-py==2.1.2.14
export LD_LIBRARY_PATH=/mnt/data/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=/mnt/data/mujoco210
$PY -c 'import mujoco_py; print("MUJOCO_PY_OK")'
$PIP install jax==0.9.2 jaxlib==0.9.2 jax-rocm7-plugin==0.9.2 jax-rocm7-pjrt==0.9.2
$PIP install 'flax==0.9.0' optax ml_collections absl-py wandb tqdm imageio moviepy 'scipy<1.14' dm_control tensorflow-cpu
$PIP install d4rl==1.1
$PIP install 'git+https://github.com/ikostrikov/dmcgym'
$PY -c 'import jax; print("JAX_DEVICES", jax.devices())'
$PY -c 'import gym, d4rl, dmcgym; print("D4RL_OK")'
echo STEP2_DONE
