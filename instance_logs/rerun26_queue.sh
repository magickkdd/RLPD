#!/bin/bash
# v26 修复版正式重跑：tanh-normal 缓存语义修复后；串行；ckpt 目录 p2_v26
cd /mnt/workspace/rlpd
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/mnt/workspace/mujoco210/bin:$LD_LIBRARY_PATH
export WANDB_MODE=offline
PY=/mnt/data/miniforge3/envs/rlpd311/bin/python
Q=/mnt/workspace/rerun26.log
run_one() {
  E=$1; S=$2
  echo "[$(date +%H:%M:%S)] START $E s$S" >> $Q
  $PY -u train_finetuning.py --env_name=$E --seed=$S \
    --start_training=5000 --max_steps=200000 --utd_ratio=20 \
    --config=configs/rlpd_config.py \
    --log_dir=/mnt/workspace/ckpt/p2_v26/RLPD/$E/s$S \
    --project_name=rlpd_p2_v26 --checkpoint_model=True \
    > /mnt/workspace/logs/v26_${E}-v0_s${S}.log 2>&1
  echo "[$(date +%H:%M:%S)] DONE_$? $E s$S" >> $Q
}
run_one halfcheetah-expert-v0 42
run_one halfcheetah-expert-v0 43
run_one halfcheetah-expert-v0 44
run_one hopper-expert-v0 42
run_one walker2d-expert-v0 42
run_one hopper-expert-v0 43
run_one hopper-expert-v0 44
run_one walker2d-expert-v0 43
run_one walker2d-expert-v0 44
run_one walker2d-medium-replay-v0 42
run_one walker2d-medium-replay-v0 43
run_one walker2d-medium-replay-v0 44
echo "[$(date +%H:%M:%S)] V26_QUEUE_ALL_DONE" >> $Q
