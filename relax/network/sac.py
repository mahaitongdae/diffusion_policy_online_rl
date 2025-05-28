from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax, jax.numpy as jnp
import haiku as hk

from relax.network.blocks import Activation, QNet, PolicyNet
from relax.network.common import WithSquashedGaussianPolicy


class SACParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    log_alpha: jax.Array


@dataclass
class SACNet(WithSquashedGaussianPolicy):
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    target_entropy: float


def create_sac_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
) -> Tuple[SACNet, SACParams]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(hk.transform(lambda obs: PolicyNet(act_dim, hidden_sizes, activation)(obs)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, policy_key = jax.random.split(key, 3)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs)
        log_alpha = jnp.array(1.0, dtype=jnp.float32)
        return SACParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, log_alpha)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = SACNet(policy=policy.apply, q=q.apply, target_entropy=-act_dim)
    return net, params
