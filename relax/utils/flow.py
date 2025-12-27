from typing import Protocol, Tuple
from dataclasses import dataclass

import numpy as np
import jax, jax.numpy as jnp
import optax

class FlowModel(Protocol):
    def __call__(self, t: jax.Array, x: jax.Array) -> jax.Array:
        ...

@dataclass(frozen=True)
class OTFlow:
    num_timesteps: int

    def p_sample(self, key: jax.Array, model: FlowModel, shape: Tuple[int, ...]) -> jax.Array:
        x = 0.5 * jax.random.normal(key, shape)
        dt = 1.0 / self.num_timesteps
        def body_fn(x, t):
            tau = t * dt
            drift = model(tau, x)
            x_next = x + drift * dt
            return x_next, None
        t_seq = jnp.arange(self.num_timesteps)
        x, _ = jax.lax.scan(body_fn, x, t_seq)
        return x

    def q_sample(self, t: int, x_start: jax.Array, noise: jax.Array):
        return t * x_start + (1 - t) * noise

    def weighted_p_loss(self, key: jax.Array, weights: jax.Array, model: FlowModel, t: jax.Array,
                        x_start: jax.Array):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        x_t = jax.vmap(self.q_sample)(t, x_start, noise)
        v_pred = model(t, x_t)
        loss = weights * optax.squared_error(v_pred, (x_start - noise))
        return loss.mean()


