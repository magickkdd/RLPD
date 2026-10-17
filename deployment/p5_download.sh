#!/bin/bash
# P5: vd4rl cheetah-run medium 64px -> persistent /mnt/workspace, symlink ~/.vd4rl
set -e
PY=/mnt/data/miniforge3/envs/rlpd311/bin
$PY/pip install -q huggingface_hub 2>&1 | tail -1 || true
mkdir -p /mnt/workspace/vd4rl_dl /root/.vd4rl
export HF_ENDPOINT=https://hf-mirror.com
$PY/hf download conglu/vd4rl --repo-type dataset \
  --include "vd4rl/main/cheetah_run/medium/64px/*" \
  --local-dir /mnt/workspace/vd4rl_dl
ln -sfn /mnt/workspace/vd4rl_dl/vd4rl/main /root/.vd4rl/main
ls /root/.vd4rl/main/cheetah_run/medium/64px/ | wc -l
$PY/python - << 'PYEOF'
import numpy as np, glob
fs = glob.glob('/root/.vd4rl/main/cheetah_run/medium/64px/*.npz')
print('npz count:', len(fs))
d = np.load(fs[0])
print('keys:', list(d.keys()))
PYEOF
echo P5_DATA_OK
