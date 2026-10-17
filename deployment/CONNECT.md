# CONNECT.md — 每次开机连接 ModelScope AMD 实例（照做即可）

> 已验证通路（2026-10-17 实测）：本机 → Tailscale tailnet（userspace）→ 实例 rlpd-mi300（100.78.127.38）→ serve 2222 → sshd 2223
> 架构依据：社区文章《在 ModelScope 免费 AMD Notebook 上搭建私有 SSH: OpenSSH + Tailscale 实战》（modelscope.cn/learn/435314），修复了其"坑 3"（双 daemon/socket 错配）——**所有 tailscale 命令必须显式 `--socket=$P/ts.sock`**

## 角色与固定资产

| 侧 | 资产 | 位置 |
|---|---|---|
| 本机 | tailscale 1.104.1（用户态，免 root） | `~/.local/ts/` |
| 本机 | tailscaled state/sock | `~/.local/share/ts/` |
| 本机 | 登录私钥 | `~/.ssh/id_ed25519`（公钥已装到实例） |
| 实例 | tailnet 设备身份 | `/mnt/workspace/.remote-access/tailscaled.state`（**持久，重启免重新授权**） |
| 实例 | sshd host key / 配置 / 公钥 | `/mnt/workspace/.remote-access/` |
| 凭据 | TS_AUTHKEY | 本机 `RLPD/deployment/.secrets.env`（authkey 仅首次入网/换网用） |
| tailnet | 实例 IP | `100.78.127.38`（固定不变，记在 `RLPD/.ts_ip`） |

## 每次开机流程（3 步，约 2 分钟）

### 第 1 步：开实例 + 打开 WebIDE 终端（唯一需要浏览器的动作）

**⚠️ v17 教训：先确认实例规格再启动！** 页面有多个实例卡片（CPU 型与 GPU 型），必须点开规格含 **192G 显存 / AMD** 字样的那台（误开 CPU 实例 = 无 GPU/ROCm，白耗一次开机）。进入后核对页面顶部规格（应为 GPU 型；若显示 `DSW- CPU` 或无 GPU 字样立即停止）。

开启 notebook 实例 → 进入 `modelscope.cn/code/workspace` → 打开底部"终端"页签。
（agent 可用 computer use 自动完成；页面冻结就刷新重开终端）

### 第 2 步：粘贴 bootstrap 一行命令（在 WebIDE 终端执行）

bootstrap 脚本已常驻持久盘，一行点火（后台运行，幂等）：

```bash
nohup bash /mnt/workspace/.remote-access/bootstrap_ts.sh > /mnt/workspace/.remote-access/bootstrap_run.log 2>&1 & sleep 8; cat /mnt/workspace/.remote-access/receipt.txt
```

看到 `RECEIPT: IP=100.78.127.38 UP_RC=0` 即成功（tailnet 身份已持久化，**不会**再要授权链接）。
若 `bootstrap_ts.sh` 不存在（持久盘被清空的极端情况）：从私有仓库拉 bundle 重传脚本，或按 `deployment/bootstrap_ts.sh` 重建后经 WebIDE 文件上传放入 `.remote-access/`。

### 第 3 步：本机连接

```bash
bash /home/czx/RLPD/deployment/connect_notebook.sh
# 或直接:
ssh -i ~/.ssh/id_ed25519 -p 2222 \
  -o ProxyCommand="$HOME/.local/ts/tailscale --socket=$HOME/.local/share/ts/ts.sock nc %h %p" \
  root@100.78.127.38
```

连上后训练操作全部走此 SSH（tmux/nohup 长训，scp 传文件），浏览器只当看板。

## 传输文件

```bash
scp -i ~/.ssh/id_ed25519 -P 2222 \
  -o ProxyCommand="$HOME/.local/ts/tailscale --socket=$HOME/.local/share/ts/ts.sock nc %h %p" \
  本地文件 root@100.78.127.38:/mnt/workspace/
```

大文件优先走 ModelScope 仓库通道（`modelscope upload/download`），不吃 tailnet 带宽。

## 故障表

| 现象 | 原因 | 处理 |
|---|---|---|
| bootstrap 卡 `up` 60s | 控制面偶发慢 | 等待；再不行 `tailscale --socket=$P/ts.sock status` 看 Logged out |
| `up` 报 authkey 无效 | key 过期/单次用尽 | 脚本已内置兜底：持久身份直连（无需 key）；仅全新 tailnet 才需要新 key |
| ssh `Connection refused` | sshd/serve 未起 | 重跑 bootstrap；`ss -ltnp | grep 2223` |
| ssh 一直超时 | 本机 tailscaled 没跑 | `bash deployment/connect_notebook.sh` 会自动拉起 |
| jax 只有 CpuDevice / 无 rocm-smi | **开成了 CPU 实例** | 关掉重开 192G 显存的 AMD GPU 实例（NAS 数据不丢） |
| WebIDE 终端冻结 | 前端 websocket 断连 | 刷新页面重开终端（不影响实例内进程） |
| 实例 IP 变化 | tailnet 身份被清（持久盘丢失） | 看 bootstrap 日志中 `TS_IP=`，更新 `RLPD/.ts_ip` |
| 永远提示 bootstrap 在运行 | 锁 fd 被后台进程继承 | 社区坑 7；脚本已用独立日志文件规避 |

## 安全注意

- `tailscaled.state`、`ssh-host-keys/` 是凭证，**永不**上传/外发；bundle 打包排除
- `connection-receipt.json`、`bootstrap_run.log` 含 IP，不外发
- 密码登录已关闭，仅公钥
