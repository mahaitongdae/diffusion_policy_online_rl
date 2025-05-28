from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax, jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import Activation, DistributionalQNet2, DACERPolicyNet, QNet, PolicyStdNet
from relax.network.common import WithSquashedGaussianPolicy
from relax.utils.diffusion import GaussianDiffusion
from relax.utils.jax_utils import random_key_from_data
from numpyro.distributions import Normal

class Diffv3Params(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    std: hk.Params
    log_alpha: jax.Array


@dataclass
class Diffv3Net:
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    std: Callable[[hk.Params, jax.Array], jax.Array]
    num_timesteps: int
    act_dim: int
    act_batch_size: int
    target_entropy: float

    @property
    def diffusion(self) -> GaussianDiffusion:
        return GaussianDiffusion(self.num_timesteps)

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        # policy_params, log_alpha = policy_params

        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)

        key, noise_key = jax.random.split(key)
        action = self.diffusion.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
        # action = action + jax.random.normal(noise_key, action.shape) * jnp.exp(log_alpha) * 0.1
        return action.clip(-1, 1)

    def get_batch_actions(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array,
                          q_func: Callable) -> jax.Array:
        key, noise_key = jax.random.split(key)
        batch_flatten_obs = obs.repeat(self.act_batch_size, axis=0)
        batch_flatten_actions = self.get_action(key, policy_params, batch_flatten_obs)
        batch_q = q_func(batch_flatten_obs, batch_flatten_actions).reshape(-1, self.act_batch_size)
        max_q_idx = batch_q.argmax(axis=1)
        batch_action = batch_flatten_actions.reshape(obs.shape[0], -1, self.act_dim) # ?
        slice = lambda x, y: x[y]
        # action: batch_size, repeat_size, idx: batch_size
        best_action = jax.vmap(slice, (0, 0))(batch_action, max_q_idx)
        # best_action, log_p = self.get_noised_actions(noise_key, std_params, obs, best_action)
        # return best_action.clip(-1, 1), log_p
        return best_action

    def get_exploration_noise(self, key: jax.Array, std_params: hk.Params, obs: jax.Array, act: jax.Array) -> (
        jax.Array, jax.Array):
        std = jax.exp(self.std(std_params, obs))
        exploration_noise = jax.random.normal(key, act.shape) * std
        # noised_action = act + exploration_noise
        logp = Normal(jnp.zeros_like(act), std).log_prob(exploration_noise).sum(axis=-1)
        return exploration_noise, logp

    def get_noised_actions_with_logp(self, key: jax.Array, policy_params: hk.Params,
                                    std_params: hk.Params, obs: jax.Array, q_func: Callable) -> (jax.Array, jax.Array):
        key, noise_key = jax.random.split(key)
        action = self.get_batch_actions(key, policy_params, obs, q_func)
        noise, logp = self.get_exploration_noise(noise_key, std_params, obs, action)
        return (action + noise).clip(-1, 1), logp

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        key = random_key_from_data(obs)
        policy_params, log_alpha = policy_params
        log_alpha = -jnp.inf
        policy_params = (policy_params, log_alpha)
        return self.get_action(key, policy_params, obs)

    def q_evaluate(
        self, key: jax.Array, q_params: hk.Params, obs: jax.Array, act: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        q_mean, q_std = self.q(q_params, obs, act)
        z = jax.random.normal(key, q_mean.shape)
        z = jnp.clip(z, -3.0, 3.0)  # NOTE: Why not truncated normal?
        q_value = q_mean + q_std * z
        return q_mean, q_std, q_value

def create_diffv3_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
    act_batch_size: int = 4,
    ) -> Tuple[Diffv3Net, Diffv3Params]:
    # q = hk.without_apply_rng(hk.transform(lambda obs, act: DistributionalQNet2(hidden_sizes, activation)(obs, act)))
    q = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(hk.transform(lambda obs, act, t: DACERPolicyNet(diffusion_hidden_sizes, activation)(obs, act, t)))
    std = hk.without_apply_rng(hk.transform(lambda obs: PolicyStdNet(act_dim, hidden_sizes, activation)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, policy_key, std_key = jax.random.split(key, 4)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs, act, 0)
        std_params = std.init(std_key, obs)
        log_alpha = jnp.array(math.log(3), dtype=jnp.float32) # math.log(3) or math.log(5) choose one
        return Diffv3Params(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, std_params, log_alpha)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = Diffv3Net(q=q.apply, policy=policy.apply, std=std.apply, num_timesteps=num_timesteps, act_dim=act_dim,
                    target_entropy=-act_dim * 0.9, act_batch_size=act_batch_size)
    return net, params
