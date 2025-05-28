from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax, jax.numpy as jnp
import haiku as hk

from relax.network.blocks import Activation, DistributionalQNet2, PolicyNet
from relax.network.common import WithSquashedGaussianPolicy


class DSACTParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    target_policy: hk.Params
    log_alpha: jax.Array


@dataclass
class DSACTNet(WithSquashedGaussianPolicy):
    q: Callable[[hk.Params, jax.Array, jax.Array], Tuple[jax.Array, jax.Array]]
    target_entropy: float

    def q_evaluate(
        self, key: jax.Array, q_params: hk.Params, obs: jax.Array, act: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        q_mean, q_std = self.q(q_params, obs, act)
        z = jax.random.normal(key, q_mean.shape)
        z = jnp.clip(z, -3.0, 3.0)  # NOTE: Why not truncated normal?
        q_value = q_mean + q_std * z
        return q_mean, q_std, q_value

def create_dsact_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
) -> Tuple[DSACTNet, DSACTParams]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act: DistributionalQNet2(hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(hk.transform(lambda obs: PolicyNet(act_dim, hidden_sizes, activation)(obs)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, policy_key = jax.random.split(key, 3)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs)
        target_policy_params = policy_params
        log_alpha = jnp.array(1.0, dtype=jnp.float32)
        return DSACTParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = DSACTNet(policy=policy.apply, q=q.apply, target_entropy=-act_dim)
    return net, params
