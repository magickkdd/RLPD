#!/bin/bash
# master 幂等启动器（守卫用文件内 bracket 模式，免疫外层引号）
if pgrep -f "[w]2_master.sh" > /dev/null 2>&1; then
  echo "MASTER_ALREADY_RUNNING $(pgrep -f '[w]2_master.sh' | head -1)"
else
  nohup setsid bash /mnt/workspace/w2_master.sh > /dev/null 2>&1 &
  sleep 2
  echo "MASTER_STARTED $(pgrep -f '[w]2_master.sh' | head -1)"
fi
