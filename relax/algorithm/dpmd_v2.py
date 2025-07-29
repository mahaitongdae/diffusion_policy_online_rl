from typing import NamedTuple, Tuple
from functools import partial

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle

from relax.algorithm.base import Algorithm
from relax.network.dacer import DACERNet, DACERParams
from relax.network.diffv2 import Diffv2Net, Diffv2Params
from relax.network.diffv4 import Diffv4Net
from relax.utils.experience import Experience
from relax.utils.typing_utils import Metric


class Diffv2OptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class Diffv2TrainState(NamedTuple):
    params: Diffv2Params
    opt_state: Diffv2OptStates
    step: int
    entropy: float
    running_mean: float
    running_std: float

class DPMDV2(Algorithm):

    def __init__(
        self,
        agent: Diffv4Net,
        params: Diffv2Params,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        alpha_lr: float = 3e-2,
        lr_schedule_end: float = 5e-5,
        tau: float = 0.005,
        delay_alpha_update: int = 250,
        delay_update: int = 2,
        reward_scale: float = 0.2,
        num_samples: int = 200,
        use_ema: bool = True,
        reweight_type: str = 'logsumexp',  # 'exp', 'square'
        learnable_alpha: bool = True,
        kl_constraint: float = 0.1,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.num_samples = num_samples
        self.optim = optax.adam(lr)
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4),
        )
        self.policy_optim = optax.adam(learning_rate=lr_schedule)
        self.alpha_optim = optax.adam(alpha_lr)
        self.entropy = 0.0
        self.reweight_type = reweight_type
        self.learnable_alpha = learnable_alpha
        self.kl_constraint = kl_constraint

        self.state = Diffv2TrainState(
            params=params,
            opt_state=Diffv2OptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                # policy=self.optim.init(params.policy),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
            running_mean=jnp.float32(0.0),
            running_std=jnp.float32(1.0)
        )
        self.use_ema = use_ema

        @jax.jit
        def stateless_update(
            key: jax.Array, state: Diffv2TrainState, data: Experience
        ) -> Tuple[Diffv2OptStates, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state = state.opt_state
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std
            next_eval_key, new_eval_key, new_q1_eval_key, new_q2_eval_key, log_alpha_key, diffusion_time_key, diffusion_noise_key = jax.random.split(
                key, 7)

            reward *= self.reward_scale

            def get_min_q(s, a):
                q1 = self.agent.q(q1_params, s, a)
                q2 = self.agent.q(q2_params, s, a)
                q = jnp.minimum(q1, q2)
                return q

            def get_min_taret_q(s, a):
                q1 = self.agent.q(target_q1_params, s, a)
                q2 = self.agent.q(target_q2_params, s, a)
                q = jnp.minimum(q1, q2)
                return q

            next_action = self.agent.get_action(next_eval_key, (policy_params, log_alpha, q1_params, q2_params), next_obs)
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(q1_target, q2_target)  # - jnp.exp(log_alpha) * next_logp
            q_backup = reward + (1 - done) * self.gamma * q_target

            def q_loss_fn(q_params: hk.Params) -> jax.Array:
                q = self.agent.q(q_params, obs, action)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss, q

            (q1_loss, q1), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)
            
            batch_action, q_batch_action = self.agent.get_batch_action_with_q(
                new_eval_key, (target_policy_params, log_alpha, target_q1_params, target_q2_params), obs
                )


            def policy_loss_fn(policy_params) -> jax.Array:
                # q_min = get_min_q(next_obs, next_action)
                # q_mean, q_std = q_min.mean(), q_min.std()
                # if self.reweight_type == 'square':
                #     q_weights = jax.nn.relu(q_min) ** 2
                #     scaled_q = q_min
                # elif self.reweight_type == 'exp':
                #     norm_q = q_min - running_mean / running_std
                #     scaled_q = norm_q.clip(-3., 3.) / jnp.exp(log_alpha)
                #     q_weights = jnp.exp(scaled_q)
                if self.reweight_type == 'logsumexp':
                    scaled_q = q_batch_action / jax.nn.softplus(log_alpha)
                    Z = jax.nn.logsumexp(scaled_q, axis=0, keepdims=True)
                    q_weights = jnp.exp(scaled_q - Z)  # [N, B, 1]
                    q_mean = jnp.mean(q_batch_action)
                    q_std = jnp.std(q_batch_action, axis=0).mean()
                else:
                    raise NotImplementedError(f"Reweight type {self.reweight_type} is not implemented.")
                def denoiser(t, x):
                    return self.agent.policy(policy_params, obs, x, t)
                if self.agent.use_flow:
                    t = jax.random.uniform(diffusion_time_key, (self.agent.num_particles, obs.shape[0],))
                else:
                    t = jax.random.randint(diffusion_time_key, (self.agent.num_particles, obs.shape[0],), 0, self.agent.num_timesteps)
                    
                loss_fn = partial(self.agent.diffusion.weighted_p_loss, key=diffusion_noise_key, model=denoiser)
                loss = jax.vmap(loss_fn)(
                    weights=jax.lax.stop_gradient(q_weights), 
                    t=t, 
                    x_start=jax.lax.stop_gradient(batch_action))
                loss = jnp.mean(loss)
                return loss, (q_weights, scaled_q, q_mean, q_std, Z)

            (total_loss, (q_weights, scaled_q, q_mean, q_std, Z)), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)

            # update alpha
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                # approx_entropy = 0.5 * self.agent.act_dim * jnp.log( 2 * jnp.pi * jnp.exp(1) * (jnp.exp(log_alpha)) ** 2)
                # kl_upper_bound = Z.squeeze(axis=0) - jnp.log(self.num_samples)
                
                log_alpha_loss = jax.nn.softplus(log_alpha) * (self.kl_constraint + Z.squeeze(axis=0) - jnp.log(self.num_samples))
                return log_alpha_loss

            # update networks
            def param_update(optim, params, grads, opt_state):
                update, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            def delay_param_update(optim, params, grads, opt_state):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda params, opt_state: param_update(optim, params, grads, opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def delay_alpha_param_update(optim, params, opt_state):
                return jax.lax.cond(
                    step % self.delay_alpha_update == 0,
                    lambda params, opt_state: param_update(optim, params, jax.grad(log_alpha_loss_fn)(params), opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def delay_target_update(params, target_params, tau):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda target_params: optax.incremental_update(params, target_params, tau),
                    lambda target_params: target_params,
                    target_params
                )

            q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)
            policy_params, policy_opt_state = delay_param_update(self.policy_optim, policy_params, policy_grads, policy_opt_state)
            if self.learnable_alpha:
                log_alpha, log_alpha_opt_state = delay_alpha_param_update(self.alpha_optim, log_alpha, log_alpha_opt_state)
            else:
                pass

            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)
            target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)

            new_running_mean = running_mean + 0.001 * (q_mean - running_mean)
            new_running_std = running_std + 0.001 * (q_std - running_std)

            state = Diffv2TrainState(
                params=Diffv2Params(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha),
                opt_state=Diffv2OptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state, log_alpha=log_alpha_opt_state),
                step=step + 1,
                entropy=jnp.float32(0.0),
                running_mean=new_running_mean,
                running_std=new_running_std
            )
            info = {
                "q1_loss": q1_loss,
                "q1_mean": jnp.mean(q1),
                "q1_max": jnp.max(q1),
                "q1_min": jnp.min(q1),
                "q2_loss": q2_loss,
                "policy_loss": total_loss,
                "alpha": jax.nn.softplus(log_alpha),
                "q_weights_std": jnp.std(q_weights),
                "q_weights_mean": jnp.mean(q_weights),
                "q_weights_min_min": jnp.min(q_weights),
                "q_weights_min_mean": jnp.min(q_weights, axis=0).mean(),
                "q_weights_max_mean": jnp.max(q_weights, axis=0).mean(),
                "q_weights_std_mean": jnp.std(q_weights, axis=0).mean(),
                "scale_q_mean": jnp.mean(scaled_q),
                "scale_q_std": jnp.std(scaled_q, axis=0).mean(),
                "scale_q_gap_mean": (jnp.max(scaled_q, axis=0) - jnp.min(scaled_q, axis=0)).mean(),
                "running_q_mean": new_running_mean,
                "running_q_std": new_running_std,
                "approx_kl": jnp.mean(q_weights * (scaled_q - Z))
                    # 0.5 * self.agent.act_dim * jnp.log( 2 * jnp.pi * jnp.exp(1) * (jnp.exp(log_alpha)) ** 2),
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)

    def get_policy_params(self):
        return (self.state.params.policy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2 )

    def get_policy_params_to_save(self):
        return (self.state.params.target_poicy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)