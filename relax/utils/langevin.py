from typing import Protocol, Tuple
from dataclasses import dataclass

import jax, jax.numpy as jnp

class ScoreModel(Protocol):
    def __call__(self, x: jax.Array) -> jax.Array:
        ...

@dataclass(frozen=True)
class LangevinDynamics:
    num_timesteps: int

    def sample(self, key: jax.Array, model: ScoreModel, shape: Tuple[int, ...]) -> jax.Array:
        x_key, noise_key = jax.random.split(key)

        x = jax.random.normal(x_key, shape)
        noise = jax.random.normal(noise_key, (self.num_timesteps, *shape))

        # Bayesian Learning via Stochastic Gradient Langevin Dynamics
        # delta = a * (b + t) ^ (-gamma),  0.5 < gamma <= 1
        t = jnp.arange(self.num_timesteps)
        # a, b, gamma = 1., 1., 1.
        # delta = a * (b + t) ** -gamma
        delta = 1 - t/self.num_timesteps

        def body_fn(x, input):
            delta, noise = input
            step = model(x)
            x += delta / 2 * step + jnp.sqrt(delta) * noise
            x = jnp.clip(x, a_min=-1.0, a_max=1.0)
            return x, None

        # TODO: need of clip or squash?
        x, _ = jax.lax.scan(body_fn, x, (delta, noise))
        return x
