# NB1 草稿：老版 JAX 强化学习训练栈在 AMD MI300X (ROCm 7.2.3) 上的完整适配实践

> 用途: ModelScope 开发者激励计划 Notebook 提交材料（标签"AMD GPU 激励计划"，+50h）
> 审核要求: 全部 cell 可跑通 + 注释完整 + 环境说明 —— **W1 实例可用时逐 cell 执行验证后导出 .ipynb 提交**
> 状态: 草稿 v1（2026-10-17）

## Notebook 结构（建议 11 cells）

### 标题 Cell（Markdown）
- 标题：在 AMD MI300X 上复活 2021 年的 JAX 强化学习代码栈：JAX 0.9 ROCm 适配 + tfp 自实现替代
- 简介：RLPD（ICML 2023）官方仓库基于 JAX 0.3.x + tensorflow-probability 老栈，与 ROCm 新版插件存在大量兼容问题。本 Notebook 给出零算法改动的完整适配路径，并附数值对拍证明等价性。

### Cell 1 — 环境自检（code）
```python
import sys, subprocess
print("Python:", sys.version)
!pip list | grep -E "^(jax|jaxlib|jax-rocm|flax|optax|numpy)" 
import jax
print("JAX:", jax.__version__, "| devices:", jax.devices())
# 期望输出: [RocmDevice(id=0)]（MI300X, gfx942）
```

### Cell 2 — 版本矩阵说明（Markdown）
- 表格：组件 | 版本 | 为什么
- jax/jaxlib/jax-rocm7-plugin/jax-rocm7-pjrt 全部 0.9.2（PyPI 四件套**同版本 pin**——混版本会静默退回 CPU）
- flax 0.10.6（0.9.0 太老：EvalTrace/level、partial.init 均挂）
- optax 0.2.4
- numpy 2.2.6（jax 0.9 需 StringDType ⇒ 必须 numpy≥2）
- gym 0.23.1 + mujoco-py 2.1.2.14（老栈）
- d4rl 1.1（PyPI 版；git master 有 python_requires<3.11 会拒装）
- Python 3.11（jax 0.9+ 要求 ≥3.11 ∩ mujoco-py 要求 ≤3.11，唯一交集）

### Cell 3 — 安装命令（code，加注释）
```bash
# JAX ROCm 四件套（勿用 GitHub release wheel——0.8.x 已从官方删档）
pip install jax==0.9.2 jaxlib==0.9.2 jax-rocm7-plugin==0.9.2 jax-rocm7-pjrt==0.9.2
# 神经网络栈
pip install flax==0.10.6 optax==0.2.4 ml_collections absl-py wandb tqdm
# 老 RL 环境（mujoco-py 构建链先行）
pip install 'setuptools<66' 'cython<3' 'numpy==2.2.6' wheel
pip install gym==0.23.1 mujoco-py==2.1.2.14
pip install d4rl==1.1
```

### Cell 4 — mujoco210 安装与 mujoco-py numpy2 重编（code）
```bash
# mujoco210 二进制（mujoco-py 2.x 只支持它）
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P /root/
tar -xzf /root/mujoco210-linux-x86_64.tar.gz -C /root/.mujoco/
apt-get install -y libgl1-mesa-dev libglew-dev libosmesa6-dev libegl1 patchelf
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
# 关键坑：numpy 2.x 下 mujoco-py 的 cymj 必须重编
rm -rf site-packages/mujoco_py/generated/* /root/.cache/mujoco_py
python -c "import mujoco_py"   # 触发重编（首次 ~2 分钟）
```

### Cell 5 — 核心适配：tfp → 纯 JAX tanh-normal 分布（code，~50 行自实现）
说明：tensorflow-probability 与新 jax 不兼容；tanh-normal 分布是 SAC 类算法的采样核心。
给出 MultivariateNormalDiag + TanhTransformedDistribution 的纯 JAX 实现（log_prob 用Change of Variables，含 clip 尾部处理）。
（代码体：从 rlpd/distributions/tanh_transformed.py 摘录）

### Cell 6 — 数值对拍验证（code）
在 CPU 参考环境（jax 0.3.25 + tfp 0.19）预生成参考值存 .npz，本 cell 加载比对：
- log_prob 网格 max diff ≈ 1e-6 ✓
- mode/采样统计矩 ✓
- 尾部行为差异说明（tfp 给 -inf/NaN，自实现 clip 到 1±1e-6 更稳）

### Cell 7 — XLA/ROCm 编译冒烟（code）
```python
import jax, jax.numpy as jnp, flax.linen as nn, optax
# 256x256 MLP + LayerNorm + Adam，jit 训练 3 步
# 验证 XLA ROCm 编译路径
```

### Cell 8 — mujoco-py + gym 环境冒烟（code）
HalfCheetah-v2 step/reset（mujoco-py numpy2 重编后的产物验证）

### Cell 9 — 完整 agent.update 冒烟（code）
rlpd SACLearner（E=10 集成 critic）构建 + replay buffer 采样 + update 一步
打印 critic_loss（预期 40~80 区间，随机初始化）

### Cell 10 — 踩坑记录（Markdown 表）
| 坑 | 症状 | 解法 |
|---|---|---|
| 四件套混版本 | 静默退回 CPU | 同版本 pin |
| numpy ABI | dtype size 96 vs 88 | 删 cymj 缓存重编 |
| flax 0.9 + jax 0.9 | EvalTrace 无 level | 升 0.10.6 |
| d4rl 注册表空 | 只剩 bullet 环境 | 缺隐藏依赖 mjrl |
| jaxlib 0.3.x 下架 | pip 装不到 | GCS 归档源（参考环境） |

### Cell 11 — 性能小结（Markdown）
- MI300X 单卡 vs A100 级的吞吐对比（引用 P1 实测：UTD20 33ms/env-step）
- 本适配已支撑 250k 步级完整 RL 训练（RLPD 复现项目进行中）

## 待办（W1 实例验证后）
1. Cell 逐个在实例上执行，修正环境差异
2. 导出 .ipynb（含输出）上传 Gallery，标签"AMD GPU 激励计划"
3. 同步提交文1（踩坑文章，+25h）——可从本 Notebook 踩坑表扩写
