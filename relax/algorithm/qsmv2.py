from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import optax
import haiku as hk

from relax.algorithm.base import Algorithm
from relax.network.qsm import QSMNet, QSMParams
from relax.utils.experience import Experience
from relax.utils.typing_utils import Metric


class QSMOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    q_score: optax.OptState

class QSMTrainState(NamedTuple):
    params: QSMParams
    opt_state: QSMOptStates


class QSM(Algorithm):
    def __init__(self, agent: QSMNet, params: QSMParams, *, gamma: float = 0.99, lr: float = 3e-4, tau: float = 0.005):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.optim = optax.adam(lr)

        self.state = QSMTrainState(
            params=params,
            opt_state=QSMOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                q_score=self.optim.init(params.q_score),
            ),
        )

        @jax.jit
        def stateless_update(
            key: jax.Array, state: QSMTrainState, data: Experience
        ) -> Tuple[QSMTrainState, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params, q_score_params = state.params
            q1_opt_state, q2_opt_state, q_score_opt_state = state.opt_state
            next_action_key = key

            # compute target q
            next_action = self.agent.get_action(next_action_key, (q_score_params, q1_params, q2_params), next_obs)
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

            # update q_score
            def q_score_loss_fn(q_score_params: hk.Params) -> jax.Array:
                q1, q1_score = self.agent.get_q_score_from_gradient(q1_params, obs, action)
                q2, q2_score = self.agent.get_q_score_from_gradient(q2_params, obs, action)
                q_score = self.agent.q_score(q_score_params, obs, action)
                # TODO: how to handle q1 and q2 here?
                q_minimum_score = jnp.where(q1.reshape(-1, 1) < q2.reshape(-1, 1), q1_score, q2_score)
                q_score_loss = jnp.mean((q_score - q_minimum_score) ** 2)
                return q_score_loss, (q1, q2)

            (q_score_loss, aux), q_score_grads = jax.value_and_grad(q_score_loss_fn, has_aux=True)(q_score_params)
            q1, q2 = aux
            q_score_update, q_score_opt_state = self.optim.update(q_score_grads, q_score_opt_state)
            q_score_params = optax.apply_updates(q_score_params, q_score_update)

            # update target q
            target_q1_params = optax.incremental_update(q1_params, target_q1_params, self.tau)
            target_q2_params = optax.incremental_update(q2_params, target_q2_params, self.tau)

            state = QSMTrainState(
                params=QSMParams(q1_params, q2_params, target_q1_params, target_q2_params, q_score_params),
                opt_state=QSMOptStates(q1_opt_state, q2_opt_state, q_score_opt_state),
            )
            info = {
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "q1": jnp.mean(q1),
                "q2": jnp.mean(q2),
                "q_score_loss": q_score_loss,
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)

    def get_policy_params(self):
        return self.state.params.q_score, self.state.params.q1, self.state.params.q2
