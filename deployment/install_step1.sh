# DEPRECATED (v17): 本脚本含过期配置（py3.12/flax0.9/tensorflow-cpu/dmcgym master/jax0.8 错误路径），勿执行。
# 恢复环境一律按 RESUME.md『恢复步骤』：requirements-rocm.txt + apt 清单 + mjrl/dmcgym 特殊装法。
#!/usr/bin/env bash
# pip step 1: build toolchain + gym + mujoco-py (system py3.12)
set -x
cd /mnt/workspace
pip install 'setuptools<66' 'cython<3' 'numpy<2' wheel
pip install gym==0.23.1
pip install mujoco-py==2.1.2.14
export LD_LIBRARY_PATH=/mnt/data/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=/mnt/data/mujoco210
python -c 'import mujoco_py; print("MUJOCO_PY_OK")'
echo STEP1_DONE
