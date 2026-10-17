#!/bin/bash
# P2~P6 数据下载（hf-mirror imone/D4RL，扁平结构——v19 核对源码后修正：本地文件名 = d4rl URL basename）
# 用法: bash download_data.sh   （在实例上执行；已下载且 h5 可读的自动跳过）
set -x
# 前置断言：~/.d4rl 必须是指向持久盘的符号链接（否则数据会落在系统盘，重启丢失）
if [ ! -L "$HOME/.d4rl" ]; then
  echo "~/.d4rl 不是符号链接——先 source env.sh 重建链接（ln -sfn /mnt/workspace/.d4rl ~/.d4rl）"
  mkdir -p /mnt/workspace/.d4rl/datasets && ln -sfn /mnt/workspace/.d4rl "$HOME/.d4rl"
elif [ "$(readlink "$HOME/.d4rl")" != "/mnt/workspace/.d4rl" ]; then
  echo "~/.d4rl 指向 $(readlink $HOME/.d4rl) 而非 /mnt/workspace/.d4rl——修正（防读空目录触发重下）"
  rm "$HOME/.d4rl" && ln -sfn /mnt/workspace/.d4rl "$HOME/.d4rl"
fi
mkdir -p ~/.d4rl/datasets
cd ~/.d4rl/datasets
BASE=https://hf-mirror.com/datasets/imone/D4RL/resolve/main
H5PY=/mnt/data/miniforge3/envs/rlpd311/bin/python
# 格式: <hf-mirror 文件名>:<d4rl URL basename 即本地文件名>
# 依据 d4rl 1.1 源码: gym_mujoco/__init__.py(L42-168) 与 locomotion/__init__.py(L325-331) 的 dataset_url basename
for pair in halfcheetah_expert-v2.hdf5:halfcheetah_expert.hdf5 \
            hopper_expert-v2.hdf5:hopper_expert.hdf5 \
            walker2d_expert-v2.hdf5:walker2d_expert.hdf5 \
            walker2d_medium_replay-v2.hdf5:walker_mixed.hdf5 \
            walker2d_medium-v2.hdf5:walker2d_medium.hdf5 \
            Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5:Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5 \
            Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5:Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5; do
  src="${pair%%:*}"; dst="${pair##*:}"
  if [ -s "$dst" ] && "$H5PY" -c "import h5py; h5py.File('$dst','r').close()" 2>/dev/null; then
    echo "SKIP $dst（已在位且 h5 可读）"
  else
    echo "DOWNLOAD $dst <- $BASE/$src"
    curl -L --fail --retry 3 -o "$dst.part" "$BASE/$src" && mv "$dst.part" "$dst" \
      && "$H5PY" -c "import h5py; f=h5py.File('$dst','r'); print('$dst OK, keys:', len(f.keys()))" \
      || echo "FAIL $dst（走本机 scp 兜底）"
  fi
done
# P4 AWAC .npy 不在此仓库（已核实 imone/D4RL 全清单无 npy）——见 P3_P7_SPEC P4 数据节三路方案
ls -la ~/.d4rl/datasets/
