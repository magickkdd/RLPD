"""P0a 冒烟测试：在 AMD 实例上逐项验证 ROCm 适配栈。

运行: python smoke_test.py
通过标准: 全部 [PASS]；任一 [FAIL] 给出下一步排查指引。
"""
import os
import shutil
import sys
import traceback

# sys.path 修复: smoke 从任意 cwd 运行均可导入 rlpd 包（报告 B3 缺陷修复）
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # /mnt/workspace（rlpd/ 的父目录）
for _p in (_ROOT, os.path.join(_ROOT, "rlpd")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

results = []


def check(name):
    def deco(fn):
        try:
            detail = fn()
            results.append((name, True, detail or ""))
        except Exception as e:
            traceback.print_exc()
            results.append((name, False, f"{type(e).__name__}: {e}"))
        return fn
    return deco


@check("1-jax-gpu")
def _():
    import jax
    devs = jax.devices()
    assert any(d.platform == "gpu" for d in devs), f"jax 未识别 GPU: {devs}"
    return f"devices={devs}"


@check("2-mlp-jit")
def _():
    # 小 MLP jit 训练一步，验证 XLA/ROCm 编译路径
    import jax, jax.numpy as jnp, flax.linen as nn, optax
    from flax.training.train_state import TrainState

    class M(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(256)(x); x = nn.LayerNorm()(x); x = nn.relu(x)
            return nn.Dense(4)(x)

    v = jnp.ones((8, 16))
    ts = TrainState.create(apply_fn=M().apply, params=M().init(jax.random.PRNGKey(0), v),
                           tx=optax.adam(1e-3))

    @jax.jit
    def step(ts, x, y):
        def loss(p):
            return ((ts.apply_fn(p, x) - y) ** 2).mean()
        g = jax.grad(loss)(ts.params)
        return ts.apply_gradients(grads=g)

    for _ in range(3):
        ts = step(ts, v, jnp.zeros((8, 4)))
    return "jit 训练 3 步 OK"


@check("3-frozendict-copy")
def _():
    # drq_learner 大量使用 copy(add_or_replace=)，确认新版 flax 仍支持
    from flax.core import FrozenDict
    fd = FrozenDict({"a": {"b": 1}})
    fd2 = fd.copy(add_or_replace={"a": {"b": 2}})
    assert fd2["a"]["b"] == 2
    return "FrozenDict.copy(add_or_replace=) 可用"


@check("4-tanh-normal")
def _():
    # 自实现分布：构建 Normal actor，验证 sample/log_prob/mode 且可 jit
    import numpy as np
    import jax, jax.numpy as jnp, flax.linen as nn
    from rlpd.distributions import TanhNormal
    from functools import partial

    class Tiny(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(6)(x)

    # TanhNormal = partial(Normal, squash_tanh=True)：验证 squash 采样域
    net = TanhNormal(base_cls=Tiny, action_dim=3)
    v = jnp.ones((4, 5))
    params = net.init(jax.random.PRNGKey(0), v)
    dist = net.apply(params, v)
    key = jax.random.PRNGKey(1)
    a = dist.sample(seed=key)
    lp = dist.log_prob(a)
    m = dist.mode()
    assert a.shape == (4, 3) and lp.shape == (4,) and m.shape == (4, 3)
    # float32 下 jnp.tanh 极端值可能出 1+ulp，放宽 1e-6
    assert np.all(np.abs(np.asarray(a)) <= 1.0 + 1e-6), "tanh 域错误"
    return f"sample/log_prob/mode OK, log_prob[0]={float(lp[0]):.3f}"


@check("5-mujoco-py")
def _():
    import mujoco_py  # noqa
    import gym
    env = gym.make("HalfCheetah-v2")
    env.reset()
    o, r, d, _ = env.step(env.action_space.sample())
    return f"HalfCheetah step OK, r={r:.3f}"


@check("6-rlpd-agent")
def _():
    # 完整 agent 构建 + 一次 update（不含 wandb/数据集）
    import numpy as np, gym
    from rlpd.agents import SACLearner
    from rlpd.data import ReplayBuffer

    env = gym.make("HalfCheetah-v2")
    kwargs = dict(num_qs=10, num_min_qs=2, critic_layer_norm=True,
                  hidden_dims=(256, 256), backup_entropy=True)
    agent = SACLearner.create(0, env.observation_space, env.action_space, **kwargs)
    rb = ReplayBuffer(env.observation_space, env.action_space, 1000)
    obs = env.reset()
    for _ in range(300):
        a = env.action_space.sample()
        nobs, r, d, _ = env.step(a)
        rb.insert(dict(observations=obs, actions=a, rewards=r, masks=1.0 - float(d),
                       dones=float(d), next_observations=nobs))
        obs = nobs if not d else env.reset()
    batch = rb.sample(256)
    agent, info = agent.update(batch, utd_ratio=2)
    return f"agent.update OK, critic_loss={float(info['critic_loss']):.3f}"


@check("7-d4rl-dataset")
def _():
    # d4rl 数据加载。数据文件由部署侧预置（hf-mirror imone/D4RL v2 数据，scp 到
    # ~/.d4rl/datasets/halfcheetah-expert-v0.h5）；用 h5path 直读，绕开 d4rl 的
    # 下载/重下逻辑（Berkeley 源在实例网络下极慢且曾被残留进程写坏）。
    import os
    import gym, d4rl  # noqa
    h5path = os.path.expanduser("~/.d4rl/datasets/halfcheetah-expert-v0.h5")
    env = gym.make("halfcheetah-expert-v0")
    ds = env.get_dataset(h5path=h5path)
    n = ds["rewards"].shape[0]
    assert n > 0
    return f"halfcheetah-expert-v0 加载 {n} 条 transitions"


@check("8-modelscope-sdk")
def _():
    # 文件通道已改走本机 API 上传，实例侧 modelscope 为可选件：缺席记 SKIP 而非 FAIL
    try:
        import modelscope  # noqa
        return f"modelscope {modelscope.__version__}"
    except ImportError:
        return "SKIP（可选，不影响训练）"


@check("10-train-import")
def _():
    # G0a 前置：train_finetuning.py 模块级可导入（覆盖 dmcgym/d4rl 等顶层依赖链）
    import importlib
    m = importlib.import_module("train_finetuning")
    return "train_finetuning 模块导入 OK"


@check("9-persist-disk")
def _():
    # 持久盘探测: 确认可写路径，供 setup_amd.sh 安装环境/数据集
    import os
    for cand in ["/mnt/data", "/root/autodl-tmp", "/workspace", "/data",
                 os.path.expanduser("~")]:
        if os.path.isdir(cand) and os.access(cand, os.W_OK):
            p = os.path.join(cand, ".rlpd_write_test")
            try:
                open(p, "w").close(); os.remove(p)
                total, _, free = shutil.disk_usage(cand)
                return f"PERSIST_ROOT={cand}, 剩余 {free/1e9:.0f}GB"
            except OSError:
                continue
    raise RuntimeError("未找到可写持久盘")


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    for name, ok, detail in results:
        if only and not any(o in name for o in only):
            continue
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print("\n===== 汇总 =====")
    fails = [n for n, ok, _ in results if not ok and (not only or any(o in n for o in only))]
    if not fails:
        print("全部通过 ✓  P0a 冒烟完成")
    else:
        print(f"失败 {len(fails)} 项: {fails}")
        print("提示: 1/2 → jax-rocm 安装与驱动; 3 → flax 版本; 4 → 适配补丁; "
              "5 → mujoco-py 编译(apt 库/LD_LIBRARY_PATH); 6 → 依赖链; "
              "7 → d4rl(网络下载, 可单独重试); 8 → pip install modelscope")
        sys.exit(1)


if __name__ == "__main__":
    main()
