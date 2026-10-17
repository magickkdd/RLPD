#!/bin/bash
# W1 一键上传全部交付物（v23：含 env.sh/requirements/smoke_test/pixels 基线；远端校验升级为 --help + bash -n）
set -e
cd "$(dirname "$0")/.."
IP=$(cat .ts_ip 2>/dev/null || echo 100.78.127.38)
SCP() {
  scp -i ~/.ssh/id_ed25519 -P 2222 \
    -o "ProxyCommand=$HOME/.local/ts/tailscale --socket=$HOME/.local/share/ts/ts.sock nc %h %p" \
    -o StrictHostKeyChecking=no "$@"
}
# 训练脚本（P2/P5 新增）
SCP rlpd/train_baselines.py         "root@$IP:/mnt/workspace/rlpd/"
SCP rlpd/final_eval.py              "root@$IP:/mnt/workspace/rlpd/"
SCP rlpd/train_baselines_pixels.py  "root@$IP:/mnt/workspace/rlpd/"
# 部署工具 + 环境与冒烟（v23：env.sh 曾漏传导致实例侧旧版把 ~/.d4rl 指回 /mnt/data）
SCP deployment/gen_runs.py deployment/aggregate.py deployment/status.sh \
    deployment/connect_notebook.sh deployment/upload_all.sh \
    "root@$IP:/mnt/workspace/deployment/"
SCP deployment/env.sh deployment/requirements-rocm.txt deployment/smoke_test.py \
    deployment/download_data.sh deployment/data_sanity.py \
    "root@$IP:/mnt/workspace/deployment/"
# 规格与文档
SCP CONNECT.md deployment/CONNECT.md deployment/RESUME.md \
    P2_SPEC.md P3_SPEC.md P4_SPEC.md P5_SPEC.md P6_SPEC.md P7_SPEC.md \
    REPRODUCTION_PLAN.md "root@$IP:/mnt/workspace/"
echo "=== 上传完成，实例侧真校验（ast.parse + 每脚本 --help 抓 unknown flag + bash -n） ==="
ssh -i ~/.ssh/id_ed25519 -p 2222 \
  -o "ProxyCommand=$HOME/.local/ts/tailscale --socket=$HOME/.local/share/ts/ts.sock nc %h %p" \
  -o StrictHostKeyChecking=no "root@$IP" '
set -e
cd /mnt/workspace/rlpd
PY=/mnt/data/miniforge3/envs/rlpd311/bin/python
for f in train_baselines.py final_eval.py train_baselines_pixels.py; do
  python3 -c "import ast; ast.parse(open(\"/mnt/workspace/rlpd/$f\").read())"
  $PY $f --help > /dev/null 2>&1 && echo "  $f --help OK" || { echo "  $f --help FAIL"; exit 1; }
done
for f in /mnt/workspace/deployment/gen_runs.py /mnt/workspace/deployment/aggregate.py /mnt/workspace/deployment/data_sanity.py; do
  python3 -c "import ast; ast.parse(open(\"$f\").read())"
done
bash -n /mnt/workspace/deployment/status.sh /mnt/workspace/deployment/download_data.sh
/mnt/data/miniforge3/envs/rlpd311/bin/python /mnt/workspace/deployment/gen_runs.py --suite p2 --dry-run > /dev/null && echo "  gen_runs 自检 OK"
echo "实例侧校验全部 OK"'
