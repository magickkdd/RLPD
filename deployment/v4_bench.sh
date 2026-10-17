#!/bin/bash
# V4 倍率重测：20k 步论文口径（eval_interval=20000 → 仅 i=0 一次 eval）
export LD_LIBRARY_PATH=/mnt/workspace/mujoco210/bin:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /mnt/workspace/rlpd
PY=/mnt/data/miniforge3/envs/rlpd311/bin/python
TS() { date +%s; }
run_one() { $PY train_finetuning.py --env_name=halfcheetah-expert-v0 --utd_ratio=20 \
  --start_training=0 --max_steps=20000 --config=configs/rlpd_config.py --eval_interval=20000 \
  --seed=$1 --log_dir=/mnt/workspace/ckpt/v4/s$1 > /mnt/workspace/logs/v4_s$1.log 2>&1; }

echo "SINGLE_A start $(TS)"; run_one 901; echo "SINGLE_A end $(TS)"
echo "SINGLE_B start $(TS)"; run_one 902; echo "SINGLE_B end $(TS)"
echo "FOUR start $(TS)"
for s in 911 912 913 914; do run_one $s & done; wait
echo "FOUR end $(TS)"
echo "SIX_A start $(TS)"
for s in 921 922 923 924 925 926; do run_one $s & done; wait
echo "SIX_A end $(TS)"
echo "SIX_B start $(TS)"
for s in 931 932 933 934 935 936; do run_one $s & done; wait
echo "SIX_B end $(TS)"
echo "V4_DONE $(TS)"
