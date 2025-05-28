from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import optax
import haiku as hk

from relax.algorithm.base import Algorithm
from relax.buffer.tree import TreeBuffer
from relax.network.dipo import DIPONet, DIPOParams
from relax.utils.experience import Experience, ObsActionPair
from relax.utils.typing_utils import Metric


class DIPOOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState


class DIPOTrainState(NamedTuple):
    params: DIPOParams
    opt_state: DIPOOptStates
    step: int


class DIPO(Algorithm):

    def __init__(
        self,
        agent: DIPONet,
        params: DIPOParams,
        diffusion_buffer: TreeBuffer,
        *,
        gamma: float = 0.99,
        lr: float = 3e-4,
        tau: float = 0.005,
        action_lr: float = 0.03,
        action_gradient_steps: int = 20,
        policy_target_delay: int = 1,
        grad_norm: float = 2.0,
        action_grad_norm: float = 0.2,
    ):
        self.agent = agent
        self.diffusion_buffer = diffusion_buffer
        self.gamma = gamma
        self.tau = tau
        self.action_gradient_steps = action_gradient_steps
        self.policy_target_delay = policy_target_delay

        self.optim = optax.adam(lr)
        self.action_optim = optax.adam(action_lr)

        if grad_norm > 0:
            self.optim = optax.chain(
                optax.clip_by_global_norm(grad_norm),
                self.optim,
            )

        if action_grad_norm > 0:
            self.action_optim = optax.chain(
                optax.clip_by_global_norm(action_grad_norm),
                self.action_optim,
            )

        self.state = DIPOTrainState(
            params=params,
            opt_state=DIPOOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                policy=self.optim.init(params.policy),
            ),
            step=jnp.int32(0),
        )

        @jax.jit
        def stateless_update(
            key: jax.Array, state: DIPOTrainState, data: Experience, diffusion_data: ObsActionPair,
        ) -> Tuple[DIPOTrainState, Metric, jax.Array]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            diffusion_obs, diffusion_action = diffusion_data.obs, diffusion_data.action
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params = state.params
            q1_opt_state, q2_opt_state, policy_opt_state = state.opt_state
            step = state.step
            next_eval_key, diffusion_time_key, diffusion_noise_key = jax.random.split(key, 3)
            # jax why:
            # If you need a new random number, you can use jax.random.split() to generate new subkeys:
            # https://jax.readthedocs.io/en/latest/jax.random.html#module-jax.random

            # compute target q
            next_action = self.agent.get_action(next_eval_key, target_policy_params, next_obs)
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(q1_target, q2_target)
            q_backup = reward + (1 - done) * self.gamma * q_target

            # update q
            def q_loss_fn(q_params: hk.Params) -> jax.Array:
                q = self.agent.q(q_params, obs, action)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss

            q1_loss, q1_grads = jax.value_and_grad(q_loss_fn)(q1_params)
            q2_loss, q2_grads = jax.value_and_grad(q_loss_fn)(q2_params)
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)

            # action gradient
            def action_update_step(action: jax.Array, action_opt_state: optax.OptState) -> jax.Array:
                def action_value_loss(action: jax.Array) -> jax.Array:
                    q1 = self.agent.q(q1_params, diffusion_obs, action)
                    q2 = self.agent.q(q2_params, diffusion_obs, action)
                    q = jnp.minimum(q1, q2)
                    return -q.sum()  # Maximize Q == Minimize -Q
                action_grads = jax.grad(action_value_loss)(action)
                action_update, action_opt_state = self.action_optim.update(action_grads, action_opt_state)
                action = optax.apply_updates(action, action_update)
                action = jnp.clip(action, -1, 1)
                return action, action_opt_state

            (diffusion_action, _), _ = jax.lax.scan(
                lambda carry, _: (action_update_step(*carry), None),
                (diffusion_action, self.action_optim.init(diffusion_action)),
                None, length=self.action_gradient_steps,
            )

            # update policy
            def policy_loss_fn(policy_params: hk.Params) -> jax.Array:
                def denoiser(t, x):
                    return self.agent.policy(policy_params, diffusion_obs, x, t)
                t = jax.random.randint(diffusion_time_key, (diffusion_obs.shape[0],), 0, self.agent.num_timesteps)
                return self.agent.diffusion.p_loss(diffusion_noise_key, denoiser, t, diffusion_action)

            policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_params)
            policy_update, policy_opt_state = self.optim.update(policy_grads, policy_opt_state)
            policy_params = optax.apply_updates(policy_params, policy_update)

            # update target q
            target_q1_params = optax.incremental_update(q1_params, target_q1_params, self.tau)
            target_q2_params = optax.incremental_update(q2_params, target_q2_params, self.tau)

            # update target policy
            target_policy_params = jax.lax.cond(
                step % self.policy_target_delay == 0,
                lambda target_policy_params: optax.incremental_update(policy_params, target_policy_params, self.tau),
                lambda target_policy_params: target_policy_params,
                target_policy_params,
            )

            state = DIPOTrainState(
                params=DIPOParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params),
                opt_state=DIPOOptStates(q1_opt_state, q2_opt_state, policy_opt_state),
                step=step + 1,
            )
            info = {
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "policy_loss": policy_loss,
            }
            return state, info, diffusion_action

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)

    def update(self, key: jax.Array, data: Experience) -> Metric:
        diffusion_data, indices = self.diffusion_buffer.sample_with_indices(data.batch_size())
        self.state, info, new_diffusion_action = self._update(key, self.state, data, diffusion_data)
        new_diffusion_data = ObsActionPair(None, new_diffusion_action)
        self.diffusion_buffer.replace(indices, new_diffusion_data)
        return {k: float(v) for k, v in info.items()}

    def warmup(self, data: Experience) -> None:
        key = jax.random.key(0)
        diffusion_data = ObsActionPair(data.obs, data.action)
        obs = data.obs[0]
        policy_params = self.get_policy_params()
        self._update(key, self.state, data, diffusion_data)
        self._get_action(key, policy_params, obs)
        self._get_deterministic_action(policy_params, obs)
