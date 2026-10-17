#!/usr/bin/env bash
# 本地一键连接 ModelScope AMD 实例（tailnet）
# 用法: bash ~/.local/ts/connect.sh   或   bash /home/czx/RLPD/deployment/connect_notebook.sh
set -e
TS=~/.local/ts/tailscale
SOCK=$HOME/.local/share/ts/ts.sock

# 1) 本地 userspace tailscaled（若未运行）
if ! $TS --socket=$SOCK status >/dev/null 2>&1; then
    nohup ~/.local/ts/tailscaled --tun=userspace-networking \
        --state=$HOME/.local/share/ts/ts.state --socket=$SOCK \
        > $HOME/.local/share/ts/tailscaled.log 2>&1 &
    sleep 4
    KEY=$(grep TS_AUTHKEY /home/czx/RLPD/deployment/.secrets.env | cut -d= -f2)
    $TS --socket=$SOCK up --authkey=$KEY --hostname=czx-dev --timeout=30s 2>/dev/null || \
        $TS --socket=$SOCK up --hostname=czx-dev --timeout=30s
fi

# 2) 实例 IP（固定 tailnet 身份，重启不变）
IP=$($TS --socket=$SOCK ip -4 >/dev/null 2>&1 && cat /home/czx/RLPD/.ts_ip 2>/dev/null || echo 100.78.127.38)

echo "连接 rlpd-mi300 ($IP:2222) ..."
exec ssh -i ~/.ssh/id_ed25519 -p 2222 \
    -o ProxyCommand="$TS --socket=$SOCK nc %h %p" \
    -o StrictHostKeyChecking=accept-new \
    root@$IP
