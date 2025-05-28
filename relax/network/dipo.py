from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Sequence, Tuple

import jax, jax.numpy as jnp
import haiku as hk

from relax.network.blocks import Activation, DiffusionPolicyNet, QNet
from relax.utils.diffusion import GaussianDiffusion
from relax.utils.jax_utils import random_key_from_data


class DIPOParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    target_policy: hk.Params


@dataclass
class DIPONet:
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    num_timesteps: int
    act_dim: int

    @property
    def diffusion(self) -> GaussianDiffusion:
        return GaussianDiffusion(self.num_timesteps)

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)

        action = self.diffusion.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
        return action.clip(-1, 1)

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        key = random_key_from_data(obs)
        return self.get_action(key, policy_params, obs)

def create_dipo_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    time_dim: int = 32,
    num_timesteps: int = 100,
) -> Tuple[DIPONet, DIPOParams]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(hk.transform(lambda obs, act, t: DiffusionPolicyNet(time_dim, hidden_sizes, activation)(obs, act, t)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, policy_key = jax.random.split(key, 3)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs, act, 0)
        target_policy_params = policy_params
        return DIPOParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = DIPONet(q=q.apply, policy=policy.apply, num_timesteps=num_timesteps, act_dim=act_dim)
    return net, params
