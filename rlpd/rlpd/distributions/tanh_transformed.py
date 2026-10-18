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

    def sample(self, seed: jax.Array) -> jnp.ndarray:
        return jnp.tanh(self.distribution.sample(seed))

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        # atanh 在 |value|→1 处发散；采样域内 |x|<1，clip 仅为数值安全
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
