#!/bin/bash
# W2 主控守护 v2：波次循环（替代 w2_refeed.sh）
# - 每 5 分钟巡检：无训练进程且未全完成 → 重建未完成清单 → 4 路开新波次
# - 崩溃自愈 + 实例重启后自动拉起（配合 bootstrap_ts.sh）二合一
# - 完成判据: log 含 "Find logs at"（wandb finish 尾行, segfault 不会打印）
LOG=/mnt/workspace/logs/w2_master.log
cd /mnt/workspace
echo "$(date +%s) MASTER_START pid=$$" >> "$LOG"
while true; do
  NPROC=$(pgrep -fc "python train_(finetuning|baselines)\.py" 2>/dev/null) || NPROC=0
  if [ "${NPROC:-0}" -eq 0 ]; then
    > runs_p2_todo.txt
    while IFS= read -r line; do
      log=$(echo "$line" | grep -oE 'p2_[A-Za-z0-9._-]+\.log' | head -1)
      if ! grep -q "Find logs at" "/mnt/workspace/logs/$log" 2>/dev/null; then
        echo "$line" >> runs_p2_todo.txt
      fi
    done < runs_p2.txt
    N=$(wc -l < runs_p2_todo.txt)
    if [ "$N" -eq 0 ]; then
      echo "W2_ALL_COMPLETE $(date +%s)" >> "$LOG"
      exit 0
    fi
    echo "$(date +%s) WAVE_START n=$N" >> "$LOG"
    xargs -P 4 -d '\n' -I CMD bash -c CMD < runs_p2_todo.txt
    echo "$(date +%s) WAVE_END" >> "$LOG"
    sleep 10
  else
    sleep 300
  fi
done
