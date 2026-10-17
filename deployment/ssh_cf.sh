#!/usr/bin/env bash
# SSH over Cloudflare quick tunnel（无需账号）
set -x
pkill -9 tmate 2>/dev/null
which sshd || apt-get install -y openssh-server
mkdir -p /run/sshd
echo 'root:rlpd2026' | chpasswd
sed -i 's/#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
/usr/sbin/sshd -p 2222
echo SSHD_UP
curl -L --http1.1 -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
dpkg -i /tmp/cloudflared.deb || apt-get install -y -f
nohup cloudflared tunnel --url tcp://localhost:2222 --protocol http2 > /tmp/cf.log 2>&1 &
sleep 15
URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/cf.log | head -1)
if [ -z "$URL" ]; then
  pkill -f 'cloudflared tunnel'
  nohup cloudflared tunnel --url tcp://localhost:2222 > /tmp/cf.log 2>&1 &
  sleep 15
  URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/cf.log | head -1)
fi
echo CF_URL=$URL
