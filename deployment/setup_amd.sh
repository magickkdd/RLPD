# DEPRECATED (v17): 本脚本含过期配置（py3.12/flax0.9/tensorflow-cpu/dmcgym master/jax0.8 错误路径），勿执行。
# 恢复环境一律按 RESUME.md『恢复步骤』：requirements-rocm.txt + apt 清单 + mjrl/dmcgym 特殊装法。
#!/usr/bin/env bash
# RLPD 复现 · ModelScope AMD 实例一键 bootstrap（幂等 + 持久盘感知，可重复执行）
#
# 持久化策略（应对 notebook 重启后系统盘清空）:
#   - conda/miniforge、mujoco210、数据集、Tailscale 状态、检查点 → 全部放持久盘
#   - $HOME 下仅放符号链接，代码按默认路径无感访问
#   - apt 系统库在系统盘，每次重启需重装（脚本内置，约 1~2 分钟）
#   - 重启后快速恢复: bash setup_amd.sh（已装项自动跳过）
#
# 用法: bash setup_amd.sh          # 完整/增量 bootstrap
#       source env.sh              # 重启后若环境已装好，仅激活+拉起服务
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUDO="$(command -v sudo || true)"
FAILED_STEPS=()

step() {
    local name="$1"; shift
    echo "==== [${name}] ===="
    if "$@"; then echo "==== [${name}] OK ===="; else
        echo "!!!! [${name}] FAILED（继续后续步骤）"; FAILED_STEPS+=("$name"); fi
}

# ---- 0. 凭据（可选） ----
if [ -f "${DIR}/.secrets.env" ]; then set -a; source "${DIR}/.secrets.env"; set +a; fi

# ---- 1. 持久盘探测 ----
PERSIST_ROOT="${PERSIST_ROOT:-}"
if [ -z "$PERSIST_ROOT" ]; then
    for cand in /mnt/data /root/autodl-tmp /workspace /data; do
        if [ -d "$cand" ] && [ -w "$cand" ]; then
            # 可写测试文件验证
            if touch "$cand/.rlpd_write_test" 2>/dev/null; then
                rm -f "$cand/.rlpd_write_test"; PERSIST_ROOT="$cand"; break
            fi
        fi
    done
fi
[ -z "$PERSIST_ROOT" ] && PERSIST_ROOT="$HOME" && \
    echo "!! 未找到持久盘挂载点，降级使用 \$HOME（重启将丢环境，请确认）"
echo "PERSIST_ROOT = $PERSIST_ROOT"

# ---- 2. 系统库（系统盘，每次重启需重装） ----
step "1-apt" $SUDO apt-get update -qq
$SUDO apt-get install -y -qq \
    libgl1-mesa-dev libglew-dev libosmesa6-dev libegl1 patchelf \
    openssh-server tmux git wget unzip || true

# ---- 3. miniforge + conda env（持久盘，已装则跳过） ----
MG="$PERSIST_ROOT/miniforge3"
if [ ! -d "$MG" ]; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$MG"
fi
source "$MG/etc/profile.d/conda.sh"
conda activate "$MG/envs/rlpd" 2>/dev/null || conda create -y -p "$MG/envs/rlpd" python=3.10
conda activate "$MG/envs/rlpd"

# ---- 4. jax-rocm + python 依赖（已装则跳过） ----
if ! python -c "import jax, flax, optax" 2>/dev/null; then
    python -c "import jaxlib" 2>/dev/null || \
    pip install --no-cache-dir https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.8.2/jaxlib-0.8.2-cp310-cp310-manylinux2014_x86_64.whl
    pip install --no-cache-dir "jax==0.8.2" jax-rocm7-pjrt jax-rocm7-plugin
    pip install --no-cache-dir -r "${DIR}/requirements-rocm.txt"
fi

# ---- 5. mujoco210（持久盘）+ $HOME 符号链接 ----
MJ="$PERSIST_ROOT/mujoco210"
if [ ! -d "$MJ" ]; then
    wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210.tar.gz
    tar -xzf /tmp/mujoco210.tar.gz -C /tmp && mv /tmp/mujoco210 "$MJ"
fi
ln -sfn "$MJ" "$HOME/.mujoco/mujoco210" 2>/dev/null || { mkdir -p "$HOME/.mujoco"; ln -sfn "$MJ" "$HOME/.mujoco/mujoco210"; }
mkdir -p "$PERSIST_ROOT/.d4rl" "$PERSIST_ROOT/.vd4rl"
ln -sfn "$PERSIST_ROOT/.d4rl" "$HOME/.d4rl"
ln -sfn "$PERSIST_ROOT/.vd4rl" "$HOME/.vd4rl"

# ---- 6. Tailscale + sshd（状态在持久盘，重启免重新授权） ----
if [ -n "${TS_AUTHKEY:-}" ] || [ -f "$PERSIST_ROOT/.ts/tailscaled.state" ]; then
    command -v tailscale >/dev/null || step "6-tailscale-install" bash -c "curl -fsSL https://tailscale.com/install.sh | sh"
    mkdir -p "$PERSIST_ROOT/.ts"
    (tailscaled --tun=userspace-networking --statedir="$PERSIST_ROOT/.ts" \
        --socket="$PERSIST_ROOT/.ts/tailscaled.sock" >/tmp/tailscaled.log 2>&1 &)
    sleep 3
    if [ -n "${TS_AUTHKEY:-}" ] && [ ! -f "$PERSIST_ROOT/.ts/tailscaled.state" ]; then
        tailscale --socket="$PERSIST_ROOT/.ts/tailscaled.sock" up \
            --authkey="$TS_AUTHKEY" --hostname=rlpd-amd 2>&1 | tail -2
    else
        tailscale --socket="$PERSIST_ROOT/.ts/tailscaled.sock" up --hostname=rlpd-amd 2>&1 | tail -2
    fi
    TS_IP=$(tailscale --socket="$PERSIST_ROOT/.ts/tailscaled.sock" ip -4 2>/dev/null | head -1)
    echo "Tailscale IP: ${TS_IP:-未获取}"
    $SUDO mkdir -p /run/sshd
    (/usr/sbin/sshd -D -p 2222 >/tmp/sshd.log 2>&1 &)
    echo "SSH 入口: ssh ${USER:-root}@${TS_IP:-<tailscale-ip>} -p 2222"
else
    echo "!! 无 TS_AUTHKEY 且无历史状态，跳过 Tailscale（浏览器后备通道）"
fi

# ---- 7. 冒烟测试 ----
step "7-smoke" python "${DIR}/smoke_test.py" "$@"

echo "==================================="
if [ ${#FAILED_STEPS[@]} -eq 0 ]; then echo "bootstrap 全部通过"; else
    printf '失败步骤: %s\n' "${FAILED_STEPS[@]}"; fi
