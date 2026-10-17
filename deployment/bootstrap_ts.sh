#!/usr/bin/env bash
# Tailscale userspace SSH bootstrap（社区架构复刻，修复 socket 错配坑）
set -x
P=/mnt/workspace/.remote-access
mkdir -p $P/ssh-host-keys
chmod 700 $P
pkill -9 tmate 2>/dev/null
pkill -9 tailscaled 2>/dev/null
sleep 1

# 1) 公钥（agent 本机）
cat > $P/authorized_key.pub << 'EOF_PUB'
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPeI8GbiuJFuXYajFlzIz+WxjR8+xXDHcFYBIOgtmhYb czx@DESKTOP-E93227A
EOF_PUB
chmod 600 $P/authorized_key.pub

# 2) 标准 OpenSSH sshd @127.0.0.1:2223（host keys 落持久盘）
if [ ! -f $P/ssh-host-keys/ssh_host_ed25519_key ]; then
  ssh-keygen -t ed25519 -f $P/ssh-host-keys/ssh_host_ed25519_key -N '' -C rlpd-mi300
fi
cat > $P/sshd_config << 'EOF_SSHD'
Port 2223
ListenAddress 127.0.0.1
PermitRootLogin yes
PubkeyAuthentication yes
PasswordAuthentication no
AuthorizedKeysFile /mnt/workspace/.remote-access/authorized_key.pub
Subsystem sftp internal-sftp
HostKey /mnt/workspace/.remote-access/ssh-host-keys/ssh_host_ed25519_key
EOF_SSHD
which sshd || apt-get install -y openssh-server
mkdir -p /run/sshd
pkill -f 'sshd -f' 2>/dev/null
/usr/sbin/sshd -f $P/sshd_config
ss -ltn | grep 2223 && echo SSHD_UP

# 3) tailscale 安装（若无）
which tailscale || curl -fsSL https://tailscale.com/install.sh | sh

# 4) 唯一 daemon：userspace + 持久 state + 显式 socket
tailscaled --tun=userspace-networking --state=$P/tailscaled.state \
  --socket=$P/ts.sock > $P/tailscaled.log 2>&1 &
sleep 4
tailscale --socket=$P/ts.sock up --authkey=TSKEY_PLACEHOLDER --hostname=rlpd-mi300 --timeout=60s 2>$P/up_err.log || \
tailscale --socket=$P/ts.sock up --hostname=rlpd-mi300 --timeout=60s >> $P/up_err.log 2>&1
UP_RC=$?
echo UP_RC=$UP_RC
IP=$(tailscale --socket=$P/ts.sock ip -4 2>/dev/null)
echo TS_IP=$IP

# 5) serve: tailnet 2222 -> 127.0.0.1:2223
if [ -n "$IP" ]; then
  tailscale --socket=$P/ts.sock serve --bg --tcp=2222 tcp://127.0.0.1:2223 2>$P/serve_err.log || \
  tailscale --socket=$P/ts.sock serve --bg 2222 127.0.0.1:2223 >> $P/serve_err.log 2>&1
  tailscale --socket=$P/ts.sock serve status 2>&1
fi
echo "RECEIPT: IP=$IP UP_RC=$UP_RC $(date)" > $P/receipt.txt
cat $P/receipt.txt
echo BOOTSTRAP_DONE

# 训练波次主控自拉起（幂等；master 内部有 0 进程才开波的保护）
if ! pgrep -f w2_master.sh > /dev/null 2>&1; then
  nohup setsid bash /mnt/workspace/w2_master.sh > /dev/null 2>&1 &
  echo "W2_MASTER_STARTED"
fi
