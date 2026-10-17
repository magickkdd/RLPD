#!/bin/bash
# W2 launch (P2 36 runs) v1
set -u
cd /mnt/workspace
PY=/mnt/data/miniforge3/envs/rlpd311/bin/python

$PY deployment/gen_runs.py --suite p2 --out runs_p2.txt || { echo GEN_FAILED; exit 1; }
echo "RUNS=$(wc -l < runs_p2.txt)"

touch w2_done_per_run.txt
grep -vFf w2_done_per_run.txt runs_p2.txt > runs_p2_todo.txt || cp runs_p2.txt runs_p2_todo.txt
echo "TODO=$(wc -l < runs_p2_todo.txt)"

nohup setsid bash -c 'xargs -P 4 -d "\n" -I CMD bash -c CMD < /mnt/workspace/runs_p2_todo.txt; echo W2_XARGS_DONE $(date +%s) > /mnt/workspace/logs/w2_done.marker' > /mnt/workspace/logs/w2_xargs_driver.log 2>&1 &
echo "LAUNCHED pid=$!"
