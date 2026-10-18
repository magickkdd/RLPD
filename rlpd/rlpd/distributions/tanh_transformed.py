"""TFP-free distribution implementations (ROCm / JAX 0.8 adaptation).

替代 tensorflow_probability 的最小实现，接口与原 tfp 版本一致:
    dist.sample(seed=key), dist.log_prob(x), dist.mode()

数值行为对齐 tfd.MultivariateNormalDiag 与
tfd.TransformedDistribution(base, tfb.Tanh())，
由 deployment/test_tanh_normal.py 与老栈 (jax 0.3.x + tfp) 对拍验证。
"""

from typing import Any, Optional

import jax
import jax.numpy as jnp

_LOG_2PI = jnp.log(2.0 * jnp.pi)


class MultivariateNormalDiag:
    """对齐 tfd.MultivariateNormalDiag(loc, scale_diag) 的最小实现."""

    def __init__(self, loc: jnp.ndarray, scale_diag: jnp.ndarray):
        self.loc = loc
        self.scale_diag = scale_diag

    def sample(self, seed: jax.Array) -> jnp.ndarray:
        eps = jax.random.normal(seed, shape=self.loc.shape)
        return self.loc + self.scale_diag * eps

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        d = value.shape[-1]
        z = (value - self.loc) / self.scale_diag
        return (
            -0.5 * jnp.sum(jnp.square(z), axis=-1)
            - 0.5 * d * _LOG_2PI
            - jnp.sum(jnp.log(self.scale_diag), axis=-1)
        )

    def mode(self) -> jnp.ndarray:
        return self.loc


class TanhTransformedDistribution:
    """对齐 tfd.TransformedDistribution(base, tfb.Tanh()) 的最小实现.

    x = tanh(y), y ~ base:
        log p(x) = log p(y) - Σ_i log(1 - tanh(y_i)^2)
    数值稳定形式: log(1 - tanh(y)^2) = 2*(log 2 - y - softplus(-2y))
    """

    def __init__(self, distribution: MultivariateNormalDiag):
        self.distribution = distribution
        # tfp TransformedDistribution 的缓存语义（v26 病理修复）：sample() 后紧接着的
        # log_prob(采样值) 必须复用精确的中间 y——从 x 反推 y（atanh+clip）在饱和区
        # (σ 触底 + |μ| 偏离) 会给出 ~1e10 的 log_prob 与 inf 梯度，而缓存路径值健康、
        # 梯度有界（tools/tanh_path_ab.py 对拍实证）。此差异是 15/15 训练猝死的根因。
        self._y_cache = None

    def sample(self, seed: jax.Array) -> jnp.ndarray:
        y = self.distribution.sample(seed)
        self._y_cache = y
        return jnp.tanh(y)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        if self._y_cache is not None and value.shape == self._y_cache.shape:
            # 刚采样过的值（RLPD 的 actor-loss / backup-entropy 两条路径均为此场景）：
            # 走精确 y（对齐 tfp 缓存行为）
            y = self._y_cache
        else:
            # 外部数据：atanh 反推，clip 仅数值安全
            y = jnp.arctanh(jnp.clip(value, -1.0 + 1e-6, 1.0 - 1e-6))
        log_prob_y = self.distribution.log_prob(y)
        log_det = jnp.sum(
            2.0 * (jnp.log(2.0) - y - jax.nn.softplus(-2.0 * y)), axis=-1
        )
        return log_prob_y - log_det

    def mode(self) -> jnp.ndarray:
        return jnp.tanh(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        return {}
