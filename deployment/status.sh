#!/bin/bash
# 跑批监控（v23 参数化：覆盖全部阶段）
# 用法: bash status.sh [p2|p3|p4|p5|p6]   （默认全部阶段）
SU="${1:-p}"
echo "=== $(date '+%F %T') ==="
NP=$(pgrep -fc 'train_(finetuning|baselines)' 2>/dev/null || echo 0)
echo "--- 训练进程数: $NP"
ps -eo pid,etime,args | grep -E 'train_(finetuning|baselines)' | grep -v grep | awk '{print "  pid="$1" elapsed="$2}' | head -10
echo "--- 各阶段最近 3 个日志尾部:"
for f in $(ls -t /mnt/workspace/logs/${SU}[0-9]_*.log 2>/dev/null | head -3); do
  echo "  [$f]"; tail -2 "$f" | sed 's/^/    /'
done
echo "--- 各阶段终评完成数:"
for n in 2 3 4 5 6; do
  C=$(ls /mnt/workspace/ckpt/p$n/*/*/s*/final_eval.json /mnt/workspace/ckpt/p$n/*/*/s*/*/final_eval.json 2>/dev/null | wc -l)
  [ "$C" -gt 0 ] && echo "  p$n: $C"
done
echo "--- ckpt 体量: $(du -sh /mnt/workspace/ckpt 2>/dev/null | cut -f1)"
echo "--- 持久盘余量: $(df -h /mnt/workspace | tail -1 | awk '{print $4" free"}')"
echo "--- GPU:"
rocm-smi --showuse 2>/dev/null | tail -6 || echo "  rocm-smi 不可用"
