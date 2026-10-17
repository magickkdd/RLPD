#!/usr/bin/env bash
# 重启后的快速恢复：激活持久盘 rlpd311 环境 + mujoco/数据集符号链接
# 用法: source env.sh   （详见 RESUME.md）
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${DIR}/.secrets.env" ] && { set -a; source "${DIR}/.secrets.env"; set +a; }

PERSIST="${PERSIST_ROOT:-/mnt/data}"
ENV=/mnt/data/miniforge3/envs/rlpd311
if [ ! -x "$ENV/bin/python" ]; then
    echo "持久盘无 rlpd311 环境 → 按 RESUME.md『恢复步骤』重建（requirements-rocm.txt + apt 清单；install_step2.sh 已废弃）"
    return 1 2>/dev/null || exit 1
fi
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/workspace/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export MUJOCO_PY_MUJOCO_PATH="/mnt/workspace/mujoco210"
mkdir -p "$HOME/.mujoco"
ln -sfn /mnt/workspace/mujoco210 "$HOME/.mujoco/mujoco210"
ln -sfn /mnt/workspace/.d4rl "$HOME/.d4rl"
ln -sfn /mnt/workspace/.vd4rl "$HOME/.vd4rl" 2>/dev/null
echo "env.sh OK: $(python -V 2>&1), jax=$(python -c 'import jax; print(jax.__version__)' 2>/dev/null)"
# v17 修正: mujoco210 实测在 /mnt/workspace（原 /mnt/data 路径有歧义风险）
