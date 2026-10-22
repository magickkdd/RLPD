import json

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}

# 内嵌数据
import json as J
curves = J.load(open('/home/czx/RLPD/artifacts/v26_curves.json'))
hc = {k: v for k, v in curves.items() if 'halfcheetah' in k and ('004035' in k or '11148' in k or '14552' in k)}
import glob
fe = {}
for f in sorted(glob.glob('/home/czx/RLPD/artifacts/final_eval_v27/**/final_eval.json', recursive=True)):
    j = J.load(open(f))
    fe.setdefault(j['env_name'], []).append([j['seed'], round(j['normalized_score'], 2)])
FE_DATA = {k: sorted(v) for k, v in fe.items()}
HC_CURVES = {k.split('|')[1][-6:]: v for k, v in hc.items()}

cells = []
cells.append(md('''# RLPD 强化学习复现实战 — AMD MI300X (ROCm 7.2.3 + JAX 0.8)

**Efficient Online Reinforcement Learning Fine-Tuning with Offline Data** (Ball et al., ICML 2023) 的完整复现，
在魔搭社区免费 AMD MI300X 实例（192GB 显存）上完成 **4 任务 × 3 seeds = 12 个 run 的训练与终评**。

## 最终结果（100 episodes 终评，d4rl normalized score）

| 任务 | seed 42 | seed 43 | seed 44 | 均值 | 论文参考 |
|---|---|---|---|---|---|
| halfcheetah-expert | 99.44 | 95.88 | 104.08 | **99.80** | ~100 |
| hopper-expert | 92.72 | 113.23 | 91.74 | **99.23** | ~100 |
| walker2d-expert | 130.06 | 126.04 | 131.48 | **129.19** | ~105 |
| walker2d-medium-replay | 120.19 | 120.43 | 126.72 | **122.45** | — |

**12/12 run 全部 ≥ 90 分。** 完整代码、训练产物与排障记录：
[github.com/magickkdd/RLPD](https://github.com/magickkdd/RLPD)

本 Notebook 覆盖：环境配置 → RLPD 核心机制 → 可运行代码 → 真实训练曲线 → 终评结果 → 工程经验。'''))

cells.append(md('''## 1. 环境配置

实例：魔搭社区免费 **AMD MI300X** Notebook（192GB VRAM / 200GB RAM / 23 核，ROCm 7.2.3 预装）。

```bash
# Python 依赖（本 Notebook 的可运行部分仅需 CPU 版 JAX；GPU 版安装指令见注释）
pip install "jax[cpu]" flax optax gymnasium mujoco matplotlib numpy   # CPU 兜底版
# pip install jax[cuda] / ROCm 版：在魔搭 AMD 实例使用平台预装的 ROCm wheel 源
#   pip install jax==0.8.x --find-links <rocm-wheels-index>   （MI300X = gfx942）
```

训练完整工程（12 run）使用：JAX 0.8.x + flax + optax + d4rl 数据集（hf-mirror 下载）+ 自实现 tanh-normal 分布（替代 tensorflow-probability）。

下面的环境检测 Cell 会自动识别当前环境（GPU / CPU 均可运行，不会报错）。'''))

cells.append(code('''# 环境检测：AMD GPU（ROCm）与 JAX 后端 —— 自动降级，保证任何环境可运行
import subprocess, sys

def rocm_info():
    """读取 rocm-smi：在魔搭 MI300X 实例上返回 GPU 型号与利用率"""
    try:
        out = subprocess.run(["rocm-smi", "--showproductname", "--showuse"],
                             capture_output=True, text=True, timeout=10).stdout
        return out.strip()[:400] or "rocm-smi available (no output)"
    except Exception as e:
        return f"rocm-smi unavailable ({type(e).__name__}) — running on non-AMD host"

print("=== ROCm / GPU ===")
print(rocm_info())

print("\\n=== JAX backend ===")
try:
    import jax
    print("jax", jax.__version__)
    print("devices:", jax.devices())
    print("backend:", jax.default_backend())
except Exception as e:
    print(f"jax not installed ({e}) — Notebook 的数值演示仍可用 numpy 运行")
    # 若需要 GPU 版 JAX（AMD): pip install jax --find-links <rocm index>
'''))

cells.append(md('''## 2. RLPD 核心机制速览

RLPD = **SAC + 离线数据对称混合 + 高 UTD + 稳定化三件套**：

| 机制 | 实现 | 作用 |
|---|---|---|
| **对称采样** | 每个 batch 一半在线 replay、一半离线数据集（`offline_ratio=0.5`） | 离线数据正则化在线学习 |
| **UTD=20** | 一次环境步 → 20 次 critic 梯度更新（actor/temp 各 1 次） | 样本效率：25k 步即达论文平台 |
| **10-Q ensemble** | target 端随机抽 2/10 取 min（REDQ 式）；actor 端取 mean | 抑制 Q 过估计 |
| **LayerNorm critic** | 每个 hidden 层 pre-activation 位置 LayerNorm | 高 UTD 下切断 activation scale 漂移（论文实证的稳定关键） |
| **target_entropy = −dim/2** | RLPD 特有（SAC 标准为 −dim） | 有离线数据兜底时熵目标减半，策略更早收敛 |

终评 100 episodes 的 normalized score = `(return − random) / (expert − random) × 100`（d4rl v0 REF 表）。'''))

cells.append(md('''## 3. 核心实现（真实工程代码节选）

完整实现见 [GitHub: rlpd/agents/sac/sac_learner.py](https://github.com/magickkdd/RLPD)。以下为本次复现中承担主要计算的两段真实代码。'''))

cells.append(code('''# ---- 真实工程代码节选 1：UTD=20 的 update 循环（critic 20 次 / actor+temp 各 1 次）----
UPDATE_LOOP = """
@partial(jax.jit, static_argnames=("utd_ratio", "q_clip"))
def update(self, batch, utd_ratio, q_clip=0.0):
    new_agent = self
    for i in range(utd_ratio):                 # UTD=20：critic 每份 mini-batch 更新一次
        def slice(x):                          # tree_map 对 DatasetDict 每个叶子切 1/20
            bs = x.shape[0] // utd_ratio
            return x[bs*i : bs*(i+1)]
        mini_batch = jax.tree_util.tree_map(slice, batch)
        new_agent, _ = new_agent.update_critic(mini_batch, q_clip)
    new_agent, actor_info = new_agent.update_actor(mini_batch)   # actor 只用最后一份
    new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])
    return new_agent, {**actor_info, **critic_info, **temp_info}
"""
print(UPDATE_LOOP)

# ---- 真实工程代码节选 2：tanh-normal log_prob 的数值稳定形式 ----
TANH_NORMAL = """
# x = tanh(y), y ~ N(mu, sigma^2)
# log p(x) = log p(y) - sum_i log(1 - tanh(y_i)^2)
# 数值稳定形式: log(1 - tanh^2(y)) = 2*(log2 - y - softplus(-2y))
#   （直接算 1 - tanh^2 在 |y|>9 时灾难性消去：f32 下 tanh(y)=1.0 → log(0) = -inf）
def log_prob(self, value):
    if self._y_cache is not None and value.shape == self._y_cache.shape:
        y = self._y_cache          # tfp 缓存语义：sample() 后 log_prob() 复用精确中间量 y
    else:
        y = jnp.arctanh(jnp.clip(value, -1.0 + 1e-6, 1.0 - 1e-6))
    log_prob_y = self.distribution.log_prob(y)
    log_det = jnp.sum(2.0 * (jnp.log(2.0) - y - jax.nn.softplus(-2.0 * y)), axis=-1)
    return log_prob_y - log_det
"""
print(TANH_NORMAL)'''))

cells.append(md('''> **工程教训（本次复现最大深坑）**：用自实现分布替代 tensorflow-probability 时，tfp 的
> `TransformedDistribution` 在同一对象上 `sample()` 后 `log_prob()` 会**复用缓存的中间量 y**；
> 初版实现未复刻该语义，饱和区（σ 触底 + μ 偏离）log_prob 爆到 ~1e21、梯度 inf，
> 引发 15/15 训练猝死。修复 = 上面的 `_y_cache`。数值对拍必须覆盖**退化/饱和边界**，
> 正常区一致 ≠ 等价。详见 [TECH_DEEP_DIVE](https://github.com/magickkdd/RLPD)。'''))

cells.append(md('''## 4. 真实训练执行（MI300X 实例）

训练通过 SSH 在魔搭 MI300X 实例的终端执行（12 run 串行队列，每 run ~40 分钟）：

```bash
# 两段式训练（第一段 0→125k 停车，第二段从 ckpt 续训到 200k/250k）
python train_finetuning.py --env_name=halfcheetah-expert-v0 --seed=42 \\
    --start_training=5000 --max_steps=125000 --utd_ratio=20 \\
    --config=configs/rlpd_config.py \\
    --log_dir=/mnt/workspace/ckpt/p2_v27/RLPD/halfcheetah-expert-v0/s42 \\
    --project_name=rlpd_p2_v27 --checkpoint_model=True
# 第二段：同命令 --max_steps=200000（自动从 125k ckpt 续训）
```

实测吞吐：**单进程 ~76 steps/s**（MI300X 独占），200k 步 ≈ 40 分钟/run。
以下 Cell 内嵌真实训练曲线数据（wandb 离线文件锚点扫描提取）并可视化。'''))

curve_data_json = J.dumps(HC_CURVES)
cells.append(code('''# 真实训练曲线（内嵌自 wandb 离线文件，MI300X 实测数据）
import json, matplotlib.pyplot as plt

curve_data = json.loads(r\'''' + "'''" + curve_data_json + "'''" + r''')

fig, ax = plt.subplots(figsize=(9, 4.2))
for run_id, pts in curve_data.items():
    steps = [p[0] for p in pts]; scores = [p[1] for p in pts]
    ax.plot(steps, scores, linewidth=1.3, label=f"seed run {run_id}")
ax.axhline(90, color="red", ls="--", lw=0.8, label="G2-1 line (90)")
ax.set_xlabel("training steps"); ax.set_ylabel("normalized score")
ax.set_title("halfcheetah-expert training evaluation — MI300X real data")
ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); plt.show()
print("解读：策略在 ~30-50k 步爬升至 90+ 平台并全程保持（无策略死亡）")
'''))

cells.append(md('''## 5. 终评结果（100 episodes × 12 run）'''))

fe_json = J.dumps(FE_DATA)
cells.append(code('''# 终评分数汇总（内嵌 12 份 final_eval.json 的真实数据）
fe_data = json.loads(r\'''' + "'''" + fe_json + "'''" + r''')

fig, ax = plt.subplots(figsize=(11.5, 4.6))
x, ticks, labels = 0, [], []
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for i, env in enumerate(sorted(fe_data)):
    for seed, norm in fe_data[env]:
        ax.bar(x, norm, color=colors[i], width=0.72)
        ax.text(x, norm + 1.5, str(norm), ha="center", fontsize=8)
        labels.append(f"s{seed}"); ticks.append(x); x += 1
    ax.text(x - 1.6, 140, env.split("-")[0] + ("/mr" if "medium" in env else ""), fontsize=9, ha="center")
    x += 1.0
ax.set_xticks(ticks); ax.set_xticklabels(labels)
ax.axhline(90, color="red", ls="--", lw=0.9, label="G2-1 absolute line (90)")
ax.axhline(100, color="gray", ls=":", lw=0.8, label="paper level (100)")
ax.set_ylim(0, 150); ax.set_ylabel("normalized score (100 episodes)")
ax.legend(fontsize=8, loc="upper left")
ax.set_title("RLPD final evaluation — 12/12 runs >= 90 | MI300X - ROCm 7.2.3 - JAX 0.8", fontsize=11)
fig.tight_layout(); plt.show()

print(f"{'env':32s} {'seeds':>26s} {'mean':>7s}")
for env in sorted(fe_data):
    ms = [n for _, n in fe_data[env]]
    print(f"{env:32s} {str(ms):>26s} {sum(ms)/len(ms):7.2f}")'''))

cells.append(md('''## 6. 工程经验（完整版见 GitHub）

1. **AMD/ROCm 栈的 JAX 适配可行且值得**：官方 repo pin JAX 0.3（无 ROCm 7 wheel），升级 0.8 仅需 ~10 处 API 适配，算法数值经对拍验证一致。
2. **单进程串行优于并发**：MI300X 单进程 76 steps/s > 4 路并发聚合 54.5；且多进程同时 XLA 初始化会触发 GPU 命令队列过订阅 segfault。
3. **长训练运行的断点续训是必需品**：orbax ckpt 每 5k 步 + 自动 resume，把平台实例随时关机的损失压到 5k 步（~70 秒）。
4. **替换第三方库时警惕隐式状态语义**（本条值 2 天排障）：见第 3 节的 tanh-normal 缓存语义。
5. **评估很快**：推理-only（无梯度更新）~1500 steps/s，12 run × 100 episodes 终评仅需 12 分钟。

## 7. 资源

- 完整代码与产物：[github.com/magickkdd/RLPD](https://github.com/magickkdd/RLPD)
- RLPD 论文：[arXiv:2302.02983](https://arxiv.org/abs/2302.02983)
- 官方实现：[github.com/ikostrikov/rlpd](https://github.com/ikostrikov/rlpd)
- 数据集：[D4RL](https://github.com/Farama-Foundation/D4RL)

*环境：魔搭社区免费 MI300X Notebook · ROCm 7.2.3 · JAX 0.8.x · Python 3.11 · 训练总量 >20h*'''))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
# source 字段转 list（nbformat 规范）
for c in nb["cells"]:
    if isinstance(c["source"], str):
        c["source"] = c["source"].splitlines(keepends=True)

out = '/home/czx/RLPD/RLPD_MI300X_Reproduction.ipynb'
json.dump(nb, open(out, 'w'), ensure_ascii=False, indent=1)
print('NB_OK', out)

# 语法验证：每个 code cell 能否 compile
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        compile(src, f"cell{i}", "exec")
print('ALL_CELLS_COMPILE_OK')
