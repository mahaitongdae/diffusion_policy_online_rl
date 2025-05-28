from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import optax
import haiku as hk

from relax.algorithm.base import Algorithm
from relax.network.dsact import DSACTNet, DSACTParams
from relax.utils.experience import Experience
from relax.utils.typing_utils import Metric


class DSACTOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class DSACTTrainState(NamedTuple):
    params: DSACTParams
    opt_state: DSACTOptStates
    step: int
    mean_q1_std: float
    mean_q2_std: float

class DSACT(Algorithm):

    def __init__(
        self,
        agent: DSACTNet,
        params: DSACTParams,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        alpha_lr: float = 3e-4,
        tau: float = 0.005,
        delay_update: int = 2,
        reward_scale: float = 0.2,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.optim = optax.adam(lr)
        self.alpha_optim = optax.adam(alpha_lr)

        self.state = DSACTTrainState(
            params=params,
            opt_state=DSACTOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                policy=self.optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=jnp.int32(0),
            mean_q1_std=jnp.float32(-1.0),
            mean_q2_std=jnp.float32(-1.0),
        )

        @jax.jit
        def stateless_update(
            key: jax.Array, state: DSACTTrainState, data: Experience
        ) -> Tuple[DSACTTrainState, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state = state.opt_state
            step, mean_q1_std, mean_q2_std = state.step, state.mean_q1_std, state.mean_q2_std
            next_eval_key, new_eval_key, new_q1_eval_key, new_q2_eval_key = jax.random.split(key, 4)

            reward *= self.reward_scale

            # compute target q
            next_action, next_logp = self.agent.evaluate(next_eval_key, target_policy_params, next_obs)
            next_q1_mean, _, next_q1_sample = self.agent.q_evaluate(new_q1_eval_key, target_q1_params, next_obs, next_action)
            next_q2_mean, _, next_q2_sample = self.agent.q_evaluate(new_q2_eval_key, target_q2_params, next_obs, next_action)
            next_q_mean = jnp.minimum(next_q1_mean, next_q2_mean)
            next_q_sample = jnp.where(next_q1_mean < next_q2_mean, next_q1_sample, next_q2_sample)
            q_target = next_q_mean - jnp.exp(log_alpha) * next_logp
            q_target_sample = next_q_sample - jnp.exp(log_alpha) * next_logp
            q_backup = reward + (1 - done) * self.gamma * q_target
            q_backup_sample = reward + (1 - done) * self.gamma * q_target_sample

            # update q
            def q_loss_fn(q_params: hk.Params, mean_q_std: float) -> jax.Array:
                q_mean, q_std = self.agent.q(q_params, obs, action)
                new_mean_q_std = jnp.mean(q_std)
                mean_q_std = jax.lax.stop_gradient(
                    (mean_q_std == -1.0) * new_mean_q_std +
                    (mean_q_std != -1.0) * (self.tau * new_mean_q_std + (1 - self.tau) * mean_q_std)
                )
                q_backup_bounded = jax.lax.stop_gradient(q_mean + jnp.clip(q_backup_sample - q_mean, -3 * mean_q_std, 3 * mean_q_std))
                q_std_detach = jax.lax.stop_gradient(jnp.maximum(q_std, 0))
                epsilon = 0.1
                q_loss = -(mean_q_std ** 2 + epsilon) * jnp.mean(
                    q_mean * jax.lax.stop_gradient(q_backup - q_mean) / (q_std_detach ** 2 + epsilon) +
                    q_std * ((jax.lax.stop_gradient(q_mean) - q_backup_bounded) ** 2 - q_std_detach ** 2) / (q_std_detach ** 3 + epsilon)
                )
                return q_loss, (q_mean, q_std, mean_q_std)

            (q1_loss, (q1_mean, q1_std, mean_q1_std)), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params, mean_q1_std)
            (q2_loss, (q2_mean, q2_std, mean_q2_std)), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params, mean_q2_std)

            # update policy
            def policy_loss_fn(policy_params: hk.Params) -> jax.Array:
                new_action, new_logp = self.agent.evaluate(new_eval_key, policy_params, obs)
                q1_mean, _ = self.agent.q(q1_params, obs, new_action)
                q2_mean, _ = self.agent.q(q2_params, obs, new_action)
                q_mean = jnp.minimum(q1_mean, q2_mean)
                policy_loss = jnp.mean(jnp.exp(log_alpha) * new_logp - q_mean)
                return policy_loss, new_logp

            (policy_loss, new_logp), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)

            # update alpha
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                log_alpha_loss = -jnp.mean(log_alpha * (new_logp + self.agent.target_entropy))
                return log_alpha_loss

            log_alpha_grads = jax.grad(log_alpha_loss_fn)(log_alpha)

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

            def delay_target_update(params, target_params, tau):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda target_params: optax.incremental_update(params, target_params, tau),
                    lambda target_params: target_params,
                    target_params
                )

            q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)
            policy_params, policy_opt_state = delay_param_update(self.optim, policy_params, policy_grads, policy_opt_state)
            log_alpha, log_alpha_opt_state = delay_param_update(self.alpha_optim, log_alpha, log_alpha_grads, log_alpha_opt_state)
            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)
            target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)

            state = DSACTTrainState(
                params=DSACTParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha),
                opt_state=DSACTOptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state, log_alpha=log_alpha_opt_state),
                step=step + 1,
                mean_q1_std=mean_q1_std,
                mean_q2_std=mean_q2_std,
            )
            info = {
                "q1_loss": q1_loss,
                "q1_mean": jnp.mean(q1_mean),
                "q1_std": jnp.mean(q1_std),
                "q2_loss": q2_loss,
                "q2_mean": jnp.mean(q2_mean),
                "q2_std": jnp.mean(q2_std),
                "policy_loss": policy_loss,
                "entropy": -jnp.mean(new_logp),
                "alpha": jnp.exp(log_alpha),
                "mean_q1_std": mean_q1_std,
                "mean_q2_std": mean_q2_std,
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)
