from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple, Union

import jax, jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import Activation, DACERPolicyNet, QNet
from relax.utils.diffusion import GaussianDiffusion
from relax.utils.flow import OTFlow
from relax.utils.jax_utils import random_key_from_data

class Diffv4Params(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    target_policy: hk.Params
    log_alpha: jax.Array
    log_noise_scale: jax.Array


@dataclass
class Diffv4Net:
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    num_timesteps: int
    act_dim: int
    num_particles: int
    num_best_of_n: int
    beta_schedule_scale: float = 1.0
    beta_schedule_type: str = 'cosine'
    use_flow: bool = False

    @property
    def diffusion(self) -> GaussianDiffusion:
        if not self.use_flow:
            return GaussianDiffusion(self.num_timesteps, 
                                    self.beta_schedule_scale,
                                    self.beta_schedule_type)
        else:
            return OTFlow(self.num_timesteps,)
            

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """
        Used
        1. For get atp1 in learning q function
        2. for getting action in eval.
        
        """
        policy_params, log_noise_scale, q1_params, q2_params = policy_params

        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.diffusion.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            return act.clip(-1, 1)

        def q_fn(act: jax.Array) -> jax.Array:
            q1 = self.q(q1_params, obs, act)
            q2 = self.q(q2_params, obs, act)
            q = jnp.minimum(q1, q2)
            return q

        key, noise_key = jax.random.split(key)
        # assert self.num_particles > 1
        if self.num_best_of_n == 1:
            act = sample(key)[0]
        else:
            keys = jax.random.split(key, self.num_best_of_n)
            acts = jax.vmap(sample)(keys)
            acts = acts + jax.random.normal(noise_key, acts.shape) * jnp.exp(log_noise_scale)
            qs = jax.vmap(q_fn)(acts)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
        
        return act

    def get_batch_action_with_q(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        policy_params, log_noise_scale, q1_params, q2_params = policy_params

        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.diffusion.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            return act.clip(-1, 1)
        
        def q_fn(act: jax.Array) -> jax.Array:
            q1 = self.q(q1_params, obs, act)
            q2 = self.q(q2_params, obs, act)
            q = jnp.minimum(q1, q2)
            return q

        key, noise_key = jax.random.split(key)
        assert self.num_particles > 1
        # if self.num_particles == 1:
        #     act = sample(key)[0]
        # else:
        keys = jax.random.split(key, self.num_particles)
        acts = jax.vmap(sample)(keys)
        #     q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
        #     act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
        acts = acts + jax.random.normal(noise_key, acts.shape) * jnp.exp(log_noise_scale)
        qs = jax.vmap(q_fn)(acts)
        return acts, qs



    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        key = random_key_from_data(obs)
        policy_params, log_noise_scale, q1_params, q2_params = policy_params
        log_noise_scale = -jnp.inf
        policy_params = (policy_params, log_noise_scale, q1_params, q2_params)
        return self.get_action(key, policy_params, obs)

    def q_evaluate(
        self, key: jax.Array, q_params: hk.Params, obs: jax.Array, act: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        q_mean, q_std = self.q(q_params, obs, act)
        z = jax.random.normal(key, q_mean.shape)
        z = jnp.clip(z, -3.0, 3.0)  # NOTE: Why not truncated normal?
        q_value = q_mean + q_std * z
        return q_mean, q_std, q_value

def create_diffv4_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
    num_particles: int = 32,
    num_best_of_n: int = 32,
    beta_schedule_scale: float = 1.0,
    beta_schedule_type: str = 'cosine',
    use_flow: bool = False,
    initial_alpha: float = 1e-4,
    initial_noise_scale: float = 0.5,
    ) -> Tuple[Diffv4Net, Diffv4Params]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(hk.transform(lambda obs, act, t: DACERPolicyNet(diffusion_hidden_sizes, activation)(obs, act, t)))

    initial_log_alpha = jnp.log(initial_alpha)
    initial_log_noise_scale = jnp.log(initial_noise_scale)

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, policy_key = jax.random.split(key, 3)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs, act, 0)
        target_policy_params = policy_params
        log_alpha = jnp.array(initial_log_alpha, dtype=jnp.float32)
        log_noise_scale = jnp.array(initial_log_noise_scale, dtype=jnp.float32)
        return Diffv4Params(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha, log_noise_scale)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = Diffv4Net(q=q.apply, policy=policy.apply, num_timesteps=num_timesteps, act_dim=act_dim, 
                    num_particles=num_particles,
                    beta_schedule_scale=beta_schedule_scale, beta_schedule_type=beta_schedule_type, use_flow=use_flow, num_best_of_n=num_best_of_n,)
    return net, params
