import math
from dataclasses import dataclass
from typing import Callable, Tuple

import jax, jax.numpy as jnp
import haiku as hk
from numpyro.distributions import Normal


@dataclass
class WithSquashedGaussianPolicy:
    policy: Callable[[hk.Params, jax.Array], Tuple[jax.Array, jax.Array]]

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """for data collection"""
        mean, std = self.policy(policy_params, obs)
        z = jax.random.normal(key, mean.shape)
        act = mean + std * z
        return jnp.tanh(act)

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """for evaluation"""
        mean, _ = self.policy(policy_params, obs)
        return jnp.tanh(mean)

    def evaluate(
        self, key: jax.Array, policy_params: hk.Params, obs: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """for algorithm update"""
        mean, std = self.policy(policy_params, obs)
        z = jax.random.normal(key, mean.shape)
        act = mean + std * z
        logp = Normal(mean, std).log_prob(act) # - 2 * (math.log(2) - act - jax.nn.softplus(-2 * act))
        return jnp.tanh(act), logp.sum(axis=-1)

@dataclass
class WithSquashedDeterministicPolicy:
    policy: Callable[[hk.Params, jax.Array], jax.Array]
    preprocess: Callable[[jax.Array], jax.Array] # TODO: more general?
    exploration_noise: float

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """for data collection"""
        obs = self.preprocess(obs)
        z = self.policy(policy_params, obs)
        act = jnp.tanh(z)
        noise = jax.random.normal(key, act.shape) * self.exploration_noise
        act = jnp.clip(act + noise, -1, 1)
        return act

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """for evaluation"""
        obs = self.preprocess(obs)
        z = self.policy(policy_params, obs)
        act = jnp.tanh(z)
        return act

    def evaluate(self, policy_params: hk.Params, processed_obs: jax.Array) -> jax.Array:
        """for algorithm update"""
        z = self.policy(policy_params, processed_obs)
        act = jnp.tanh(z)
        return act

@dataclass
class WithClippedGaussianPolicy:
    policy: Callable[[hk.Params, jax.Array], Tuple[jax.Array, jax.Array]]

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """for data collection"""
        mean, std = self.policy(policy_params, obs)
        act = Normal(mean, std).sample(key)
        return act

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """for evaluation"""
        act, _ = self.policy(policy_params, obs)
        return act
