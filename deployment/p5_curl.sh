#!/bin/bash
# P5 数据直抓 v2（curl 循环, 绕过 hf CLI 的 endpoint 问题）
set -u
D=/mnt/workspace/vd4rl_dl/vd4rl/main/cheetah_run/medium/64px
mkdir -p $D
cat /mnt/workspace/p5_files.txt | xargs -P 4 -d '\n' -I F bash -c '
  f="F"; b=$(basename "$f")
  out="'"$D"'/$b"
  for i in 1 2 3 4 5; do
    curl -sL --max-time 300 -C - -o "$out" "https://hf-mirror.com/datasets/conglu/vd4rl/resolve/main/$f" && break
    sleep $((i*3))
  done
'
N=$(ls $D/*.npz 2>/dev/null | wc -l)
echo "NPZ_COUNT=$N"
ln -sfn /mnt/workspace/vd4rl_dl/vd4rl/main /root/.vd4rl/main
/mnt/data/miniforge3/envs/rlpd311/bin/python - << 'PYEOF'
import numpy as np, glob
fs = glob.glob('/root/.vd4rl/main/cheetah_run/medium/64px/*.npz')
print('npz:', len(fs))
if fs:
    d = np.load(fs[0])
    print('keys:', list(d.keys()))
PYEOF
echo P5_DATA_OK
