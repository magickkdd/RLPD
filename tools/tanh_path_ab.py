"""A/B 对拍：饱和动作区，我们的 log_prob(atanh(clip(x)) 路径) vs tfp 缓存语义(精确 y 路径)。
官方 tfp TransformedDistribution 在同一对象上 sample() 后 log_prob(x) 会复用缓存的中间 y，
梯度完整流经 softplus 项；我们的实现从 x 反推 y（atanh + clip），饱和区 clip 截断梯度。"""
import jax, jax.numpy as jnp, numpy as np

def log_prob_ours(mu, log_sig, eps):
    sig = jnp.exp(log_sig)
    x = jnp.tanh(mu + sig * eps)
    y = jnp.arctanh(jnp.clip(x, -1.0 + 1e-6, 1.0 - 1e-6))
    lp_y = -0.5 * jnp.sum(((y - mu) / sig) ** 2) - 0.5 * y.shape[-1] * np.log(2 * np.pi) - jnp.sum(jnp.log(sig))
    log_det = jnp.sum(2.0 * (jnp.log(2.0) - y - jax.nn.softplus(-2.0 * y)))
    return lp_y - log_det

def log_prob_exact(mu, log_sig, eps):
    sig = jnp.exp(log_sig)
    y = mu + sig * eps  # 缓存路径：y 精确，不做 atanh 反推
    lp_y = -0.5 * jnp.sum(((y - mu) / sig) ** 2) - 0.5 * y.shape[-1] * np.log(2 * np.pi) - jnp.sum(jnp.log(sig))
    log_det = jnp.sum(2.0 * (jnp.log(2.0) - y - jax.nn.softplus(-2.0 * y)))
    return lp_y - log_det

eps = jax.random.normal(jax.random.PRNGKey(0), (3,))
cases = [("normal mu=0 sig=1", 0.0, 0.0), ("mu=2 sig=0.5", 2.0, -0.693),
         ("mu=5 sig=1e-9(clamped)", 5.0, -20.0), ("mu=20 sig=1e-9", 20.0, -20.0),
         ("mu=-20 sig=1e-9", -20.0, -20.0), ("mu=100 sig=1e-9", 100.0, -20.0)]
print(f"{'case':28s} {'ours_value':>14s} {'exact_value':>14s} {'dval':>10s} | {'|grad_ours|':>12s} {'|grad_exact|':>12s}")
for name, mu0, ls0 in cases:
    mu = jnp.array([mu0]*3); ls = jnp.array([ls0]*3)
    va = log_prob_ours(mu, ls, eps); vb = log_prob_exact(mu, ls, eps)
    ga = jax.grad(lambda m, s: log_prob_ours(m, s, eps).sum(), argnums=(0,1))(mu, ls)
    gb = jax.grad(lambda m, s: log_prob_exact(m, s, eps).sum(), argnums=(0,1))(mu, ls)
    na = np.linalg.norm(np.concatenate([np.asarray(g).ravel() for g in ga]))
    nb = np.linalg.norm(np.concatenate([np.asarray(g).ravel() for g in gb]))
    print(f"{name:28s} {float(va):14.4f} {float(vb):14.4f} {float(va-vb):10.4f} | {na:12.3e} {nb:12.3e}")
