from typing import NamedTuple, Tuple
from functools import partial

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle

from relax.algorithm.base import Algorithm
from relax.network.diffv4 import Diffv4Net, Diffv4Params
from relax.utils.experience import Experience
from relax.utils.typing_utils import Metric


class DPMDv2OptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    alpha_variable: optax.OptState
    log_noise_scale: optax.OptState


class Diffv2TrainState(NamedTuple):
    params: Diffv4Params
    opt_state: DPMDv2OptStates
    step: int
    entropy: float
    running_mean: float
    running_std: float

def softplus_inv(x: float):
    return jnp.log(jnp.exp(x) - 1)

def solve_v_batch(x, l, lower_bound=0.0):
    """
    Solves for v such that mean(max(lower_bound, (x - v) / l)) = 1.

    Args:
        x: Input data of shape (batch_size, num_samples)
        l: Scale parameter of shape (batch_size, 1) or scalar.
        lower_bound: The clipping floor b (scalar or broadcastable).

    Returns:
        v: Solution of shape (batch_size, 1)
    """
    N = x.shape[-1]

    # Calculate the raw floor value B = l * b
    B = l * lower_bound

    # Modified target sum:
    # We transform sum(max(b, (x-v)/l)) = N into:
    # sum(ReLU(x - (v + B))) = N * l * (1 - lower_bound)
    target_sum = N * (l - B)

    # 1. Sort descending
    x_sorted = jnp.sort(x, axis=-1)[:, ::-1]

    # 2. Compute Cumulative Sums
    cumsum_x = jnp.cumsum(x_sorted, axis=-1)

    # 3. Create index array k
    k_indices = jnp.arange(1, N + 1).reshape(1, -1)

    # 4. Compute Excess Mass at the boundary (v = x_k)
    # This is the sum of (x_i - x_k) for the top k terms
    excess = cumsum_x - k_indices * x_sorted

    # 5. Determine active elements (k*)
    # We find the largest k where the available excess mass is less than the target.
    mask = excess <= target_sum
    k_star = jnp.sum(mask, axis=-1, keepdims=True)

    # 6. Solve for the shifted variable w = v + B
    # w = (sum_active - target_sum) / k_star
    sum_active = jnp.take_along_axis(cumsum_x, k_star - 1, axis=-1)
    w = (sum_active - target_sum) / k_star

    # 7. Shift back to get v
    v = w - B

    return v

def solve_v_squared_batch(x, l, lower_bound=0.0):
    """
    Solves for v such that mean((max(lower_bound, (x - v) / l))^2) = 1.

    Args:
        x: Input data of shape (batch_size, num_samples)
        l: Scale parameter of shape (batch_size, 1) or scalar.
        lower_bound: The clipping floor b.

    Returns:
        v: Solution of shape (batch_size, 1)
    """
    N = x.shape[-1]
    B = l * lower_bound

    # Target total energy: N * l^2
    C = N * (l ** 2)

    # 1. Sort descending
    x_sorted = jnp.sort(x, axis=-1)[:, ::-1]

    # 2. Cumulative sums
    cumsum_x = jnp.cumsum(x_sorted, axis=-1)
    cumsum_x2 = jnp.cumsum(x_sorted ** 2, axis=-1)
    k_indices = jnp.arange(1, N + 1).reshape(1, -1)

    # 3. Calculate "Standard" Energy and Excess at boundary x_k
    # Standard Energy: sum((x_i - x_k)^2)
    energy_std = (cumsum_x2 - 2 * x_sorted * cumsum_x + k_indices * (x_sorted ** 2))

    # Excess Mass: sum(x_i - x_k)
    excess = cumsum_x - k_indices * x_sorted

    # 4. Total Energy Check
    # At the boundary v = x_k - B, the active set is exactly 1..k.
    # The total squared error includes the active terms shifted by B and the inactive floor B.
    # E_total = E_std + 2*B*Excess + N*B^2
    energy_at_boundary = energy_std + 2 * B * excess + N * (B ** 2)

    # 5. Determine active set size k*
    # Find largest k where energy at boundary is less than C
    mask = energy_at_boundary <= C
    k_star = jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1)

    # 6. Gather statistics
    S1 = jnp.take_along_axis(cumsum_x, k_star - 1, axis=-1)
    S2 = jnp.take_along_axis(cumsum_x2, k_star - 1, axis=-1)

    # 7. Solve Quadratic
    # We solve sum((x_i - v)^2) = C_active
    # The target for the active portion is reduced by the fixed energy of the inactive floor.
    C_active = C - (N - k_star) * (B ** 2)

    # Discriminant: (2*S1)^2 - 4*k*(S2 - C_active)
    delta = S1**2 - k_star * (S2 - C_active)

    # v = (S1 - sqrt(delta)) / k
    v = (S1 - jnp.sqrt(jnp.maximum(delta, 0.0))) / k_star

    return v

class DPMDV2Fix12345BonNoiseN2(Algorithm):

    def __init__(
        self,
        agent: Diffv4Net,
        params: Diffv4Params,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        alpha_lr: float = 3e-2,
        lr_schedule_end: float = 5e-5,
        lr_schedule_steps: int = int(5e4),
        lr_schedule_begin: int = int(2.5e4),
        tau: float = 0.005,
        delay_update: int = 2,
        reward_scale: float = 0.2,
        use_ema: bool = True,
        reweight_type: str = 'logsumexp',  # 'exp', 'square'
        learnable_alpha: bool = True,
        kl_constraint: float = 0.1,
        min_alpha: float = 1e-6,
        update_additive_noise_scale: bool = True,
        initial_noise_scale: float = 0.5,
        target_noise_scale: float = 0.1,
        use_analytical_alpha_grad: bool = True,
        delay_log_noise_scale_update: int = 250,
        clipped_lower_bound: float = 0.0,
        negative_weights_regularization: float = 0.0,
        regularization_type: str = 'square',
        clipped_only_weighted_mse_lower_bound: float = -1.0,
        use_timestep_weight: bool = False,
        noise_scale_lr: float = 7e-3,
        add_state_level_reweighting: bool = False,
        bellman_next_action_use_target_policy: bool = False,
        policy_batch_action_use_target_policy: bool = True,
    ):
        self.agent = agent
        self.agent.num_best_of_n = 2  # override BON count to 2
        self.gamma = gamma
        self.tau = tau
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.optim = optax.adam(lr)
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=lr_schedule_steps,
            transition_begin=lr_schedule_begin,
        )
        self.policy_optim = optax.adam(learning_rate=lr_schedule)
        self.alpha_optim = optax.adam(alpha_lr)
        self.noise_optim = optax.adam(learning_rate=noise_scale_lr)
        self.entropy = 0.0
        self.reweight_type = reweight_type
        self.learnable_alpha = learnable_alpha
        self.kl_constraint = kl_constraint
        self.min_alpha = min_alpha
        self.target_noise_scale = target_noise_scale
        self.update_additive_noise_scale = update_additive_noise_scale
        self.use_analytical_alpha_grad = use_analytical_alpha_grad
        self.delay_log_noise_scale_update = delay_log_noise_scale_update
        self.state = Diffv2TrainState(
            params=params,
            opt_state=DPMDv2OptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                # policy=self.optim.init(params.policy),
                policy=self.policy_optim.init(params.policy),
                alpha_variable=self.alpha_optim.init(params.alpha_variable),
                log_noise_scale=self.noise_optim.init(params.log_noise_scale),
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
            running_mean=jnp.float32(0.0),
            running_std=jnp.float32(1.0)
        )
        self.use_ema = use_ema
        self.clipped_lower_bound = clipped_lower_bound
        self.negative_weights_regularization = negative_weights_regularization
        self.regularization_type = regularization_type
        self.clipped_only_weighted_mse_lower_bound = clipped_only_weighted_mse_lower_bound
        self.use_timestep_weight = use_timestep_weight
        self.add_state_level_reweighting = add_state_level_reweighting
        self.bellman_next_action_use_target_policy = bellman_next_action_use_target_policy
        self.policy_batch_action_use_target_policy = policy_batch_action_use_target_policy

        # Bug #6 fix: Monkey-patch get_action to clip actions after noise addition
        # Original code adds noise after clipping, so actions can exceed [-1, 1].
        # Fix: clip after noise, before Q evaluation.
        original_get_action = self.agent.get_action
        def fixed_get_action(key, policy_params, obs):
            from typing import Union
            policy_params_inner, log_noise_scale, q1_params, q2_params = policy_params

            def model_fn(t, x):
                return self.agent.policy(policy_params_inner, obs, x, t)

            def sample(key):
                act = self.agent.diffusion.p_sample(key, model_fn, (*obs.shape[:-1], self.agent.act_dim))
                return act.clip(-1, 1)

            def q_fn(act):
                q1 = self.agent.q(q1_params, obs, act)
                q2 = self.agent.q(q2_params, obs, act)
                q = jnp.minimum(q1, q2)
                return q

            key, noise_key = jax.random.split(key)
            if self.agent.num_best_of_n == 1:
                act = sample(key)
            else:
                keys = jax.random.split(key, self.agent.num_best_of_n)
                acts = jax.vmap(sample)(keys)
                acts = (acts + jax.random.normal(noise_key, acts.shape) * jnp.exp(log_noise_scale)).clip(-1, 1)
                qs = jax.vmap(q_fn)(acts)
                q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
                act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)

            return act

        self.agent.get_action = fixed_get_action

        # Bug #6 fix: Monkey-patch get_batch_action_with_q to clip actions after noise addition
        original_get_batch_action_with_q = self.agent.get_batch_action_with_q
        def fixed_get_batch_action_with_q(key, policy_params, obs):
            policy_params_inner, log_noise_scale, q1_params, q2_params = policy_params

            def model_fn(t, x):
                return self.agent.policy(policy_params_inner, obs, x, t)

            def sample(key):
                act = self.agent.diffusion.p_sample(key, model_fn, (*obs.shape[:-1], self.agent.act_dim))
                return act.clip(-1, 1)

            def q_fn(act):
                q1 = self.agent.q(q1_params, obs, act)
                q2 = self.agent.q(q2_params, obs, act)
                q = jnp.minimum(q1, q2)
                return q

            key, noise_key = jax.random.split(key)
            assert self.agent.num_particles > 1
            keys = jax.random.split(key, self.agent.num_particles)
            acts = jax.vmap(sample)(keys)
            acts = (acts + jax.random.normal(noise_key, acts.shape) * jnp.exp(log_noise_scale)).clip(-1, 1)
            qs = jax.vmap(q_fn)(acts)
            return acts, qs

        self.agent.get_batch_action_with_q = fixed_get_batch_action_with_q

        @jax.jit
        def stateless_update(
            key: jax.Array, state: Diffv2TrainState, data: Experience
        ) -> Tuple[DPMDv2OptStates, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, alpha_variable, log_noise_scale = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, alpha_opt_state, log_noise_scale_opt_state = state.opt_state
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std
            next_eval_key, new_eval_key, diffusion_time_key, diffusion_noise_key = jax.random.split(
                key, 4)
            # Bug #4 fix: Split noise key for each particle to get diverse gradients
            diffusion_noise_keys = jax.random.split(diffusion_noise_key, self.agent.num_particles)

            if self.learnable_alpha:
                self.alpha_transformation = "exp"
                alpha_transform_fn = jnp.exp
            else:
                self.alpha_transformation = "identity"
                alpha_transform_fn = lambda x: x
            alpha = alpha_transform_fn(alpha_variable)

            reward *= self.reward_scale

            # def get_min_q(s, a):
            #     q1 = self.agent.q(q1_params, s, a)
            #     q2 = self.agent.q(q2_params, s, a)
            #     q = jnp.minimum(q1, q2)
            #     return q

            # def get_min_taret_q(s, a):
            #     q1 = self.agent.q(target_q1_params, s, a)
            #     q2 = self.agent.q(target_q2_params, s, a)
            #     q = jnp.minimum(q1, q2)
            #     return q

            pev_policy_params = (
                target_policy_params
                if self.bellman_next_action_use_target_policy
                else policy_params
            )
            next_action = self.agent.get_action(next_eval_key, (pev_policy_params, log_noise_scale, q1_params, q2_params), next_obs)  # noise + BON, same as env interaction
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(q1_target, q2_target)
            q_backup = reward + (1 - done) * self.gamma * q_target

            def q_loss_fn(q_params: hk.Params) -> jax.Array:
                q = self.agent.q(q_params, obs, action)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss, q

            (q1_loss, q1), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)
            td_error_abs = jnp.abs(jnp.minimum(q1, q2) - q_backup)
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)

            batch_policy_params = (
                target_policy_params
                if self.policy_batch_action_use_target_policy
                else policy_params
            )
            batch_action, q_batch_action = self.agent.get_batch_action_with_q(
                new_eval_key, (batch_policy_params, log_noise_scale, target_q1_params, target_q2_params), obs
                )  # [N, B, A], [N, B]


            def policy_loss_fn(policy_params) -> jax.Array:

                assert self.reweight_type in {
                    "strictly_normalized_relu_linear",
                    "strictly_normalized_relu_square",
                    "strictly_normalized_logsumexp",
                    "negative_strictly_normalized_logsumexp",
                    "negative_strictly_normalized_relu_linear",
                }, "Unchecked reweight type"

                if self.reweight_type == 'normalized_relu_linear':
                    assert not self.learnable_alpha, "normalized_relu_linear is not compatible with learnable_alpha"
                    assert self.alpha_transformation == 'identity', "normalized_relu_linear is not compatible with alpha_transformation != identity"
                    # q_min = get_min_q(next_obs, next_action)
                    batch_q_mean, batch_q_std = q_batch_action.mean(axis=0, keepdims=True), q_batch_action.std(axis=0, keepdims=True)
                    q_normalized = (q_batch_action + alpha - batch_q_mean) / (batch_q_std + 1e-6)
                    q_weights = jax.nn.relu(q_normalized)
                    scaled_q = q_normalized
                    q_mean = batch_q_mean.mean()
                    q_std = batch_q_std.mean()
                    entropy = jax.scipy.special.entr(q_weights / q_weights.sum(axis=0, keepdims=True)).sum(axis=0)
                elif self.reweight_type == 'strictly_normalized_relu_linear':
                    normalized_diff = solve_v_batch(q_batch_action.T, alpha).T  # pass in [B, N] and get [B, 1]
                    batch_q_mean, batch_q_std = q_batch_action.mean(axis=0, keepdims=True), q_batch_action.std(axis=0, keepdims=True)
                    q_normalized = (q_batch_action - normalized_diff) / alpha
                    q_weights = jax.nn.relu(q_normalized)
                    scaled_q = q_normalized
                    q_mean = batch_q_mean.mean()
                    q_std = batch_q_std.mean()
                    entropy = jax.scipy.special.entr(q_weights / q_weights.sum(axis=0, keepdims=True)).sum(axis=0)
                elif self.reweight_type == 'negative_strictly_normalized_relu_linear':
                    # assert not self.learnable_alpha, "strictly_normalized_relu_linear is not compatible with learnable_alpha"
                    # assert clipped_lower_bound <= 0, "negative_strictly_normalized_relu_linear is not compatible with clipped_lower_bound != -jnp.inf"
                    # assert self.alpha_transformation == 'identity', "strictly_normalized_relu_linear is not compatible with alpha_transformation != identity"
                    # q_min = get_min_q(next_obs, next_action)
                    normalized_diff = solve_v_batch(q_batch_action.T, alpha, lower_bound=self.clipped_lower_bound).T  # pass in [B, N] and get [B, 1]
                    batch_q_mean, batch_q_std = q_batch_action.mean(axis=0, keepdims=True), q_batch_action.std(axis=0, keepdims=True)
                    q_normalized = (q_batch_action - normalized_diff) / alpha
                    q_weights = jnp.clip(q_normalized, clipped_lower_bound, jnp.inf)
                    scaled_q = q_normalized
                    q_mean = batch_q_mean.mean()
                    q_std = batch_q_std.mean()
                    entropy_var = jnp.clip(q_weights, 0.0, jnp.inf)
                    entropy = jax.scipy.special.entr(entropy_var / entropy_var.sum(axis=0, keepdims=True)).sum(axis=0)
                elif self.reweight_type == 'normalized_relu_square':
                    assert not self.learnable_alpha, "normalized_relu_square is not compatible with learnable_alpha"
                    assert self.alpha_transformation == 'identity', "normalized_relu_square is not compatible with alpha_transformation != identity"
                    # q_min = get_min_q(next_obs, next_action)
                    batch_q_mean, batch_q_std = q_batch_action.mean(axis=0, keepdims=True), q_batch_action.std(axis=0, keepdims=True)
                    q_normalized = (q_batch_action + alpha - batch_q_mean) / (batch_q_std + 1e-6)
                    q_weights = jax.nn.relu(q_normalized) ** 2
                    scaled_q = q_normalized
                    q_mean = batch_q_mean.mean()
                    q_std = batch_q_std.mean()
                    entropy = jax.scipy.special.entr(q_weights / q_weights.sum(axis=0, keepdims=True)).sum(axis=0)
                elif self.reweight_type == 'strictly_normalized_relu_square':
                    normalized_diff = solve_v_squared_batch(q_batch_action.T, alpha).T  # pass in [B, N] and get [B, 1]
                    batch_q_mean, batch_q_std = q_batch_action.mean(axis=0, keepdims=True), q_batch_action.std(axis=0, keepdims=True)
                    q_normalized = (q_batch_action - normalized_diff) / alpha
                    q_weights = jax.nn.relu(q_normalized) ** 2
                    scaled_q = q_normalized
                    q_mean = batch_q_mean.mean()
                    q_std = batch_q_std.mean()
                    entropy = jax.scipy.special.entr(q_weights / q_weights.sum(axis=0, keepdims=True)).sum(axis=0)
                elif self.reweight_type == 'negative_strictly_normalized_relu_square':
                    assert not self.learnable_alpha, "strictly_normalized_relu_square is not compatible with learnable_alpha"
                    assert clipped_lower_bound <= 0, "negative_strictly_normalized_relu_square is not compatible with clipped_lower_bound != -jnp.inf"
                    # assert self.alpha_transformation == 'identity', "strictly_normalized_relu_square is not compatible with alpha_transformation != identity"
                    # q_min = get_min_q(next_obs, next_action)
                    normalized_diff = solve_v_squared_batch(q_batch_action.T, alpha, lower_bound=self.clipped_lower_bound).T  # pass in [B, N] and get [B, 1]
                    batch_q_mean, batch_q_std = q_batch_action.mean(axis=0, keepdims=True), q_batch_action.std(axis=0, keepdims=True)
                    q_normalized = (q_batch_action - normalized_diff) / alpha
                    q_weights = jnp.clip(q_normalized, clipped_lower_bound, jnp.inf) ** 2
                    scaled_q = q_normalized
                    q_mean = batch_q_mean.mean()
                    q_std = batch_q_std.mean()
                    entropy_var = jnp.clip(q_weights, 0.0, jnp.inf)
                    entropy = jax.scipy.special.entr(entropy_var / entropy_var.sum(axis=0, keepdims=True)).sum(axis=0)
                elif self.reweight_type == 'normalized_leaky_relu_linear':
                    assert not self.learnable_alpha, "normalized_leaky_relu_linear is not compatible with learnable_alpha"
                    assert self.alpha_transformation == 'identity', "normalized_leaky_relu_linear is not compatible with alpha_transformation != identity"
                    # q_min = get_min_q(next_obs, next_action)
                    batch_q_mean, batch_q_std = q_batch_action.mean(axis=0, keepdims=True), q_batch_action.std(axis=0, keepdims=True)
                    q_normalized = (q_batch_action  + alpha - batch_q_mean) / (batch_q_std + 1e-6)
                    q_weights = jax.nn.leaky_relu(q_normalized)
                    scaled_q = q_normalized
                    q_mean = batch_q_mean.mean()
                    q_std = batch_q_std.mean()
                    entropy_weights = jax.nn.relu(q_normalized)
                    entropy = jax.scipy.special.entr(entropy_weights / entropy_weights.sum(axis=0, keepdims=True)).sum(axis=0) # q_batch_action [N, B]
                elif self.reweight_type == 'normalized_sigmoid_linear':
                    # assert not self.learnable_alpha, "normalized_sigmoid_linear is not compatible with learnable_alpha"
                    # assert self.alpha_transformation == 'identity', "normalized_sigmoid_linear is not compatible with alpha_transformation != identity"
                    # q_min = get_min_q(next_obs, next_action)
                    batch_q_mean, batch_q_std = q_batch_action.mean(axis=0, keepdims=True), q_batch_action.std(axis=0, keepdims=True)
                    q_normalized = (q_batch_action - batch_q_mean) / (batch_q_std + 1e-6)
                    if self.learnable_alpha:
                        q_normalized = q_normalized / alpha
                    q_weights = jax.nn.sigmoid(q_normalized)
                    scaled_q = q_normalized
                    q_mean = batch_q_mean.mean()
                    q_std = batch_q_std.mean()
                    # entropy_weights = jax.nn.sigmoid(q_normalized)
                    entropy = jax.scipy.special.entr(q_weights / q_weights.sum(axis=0, keepdims=True)).sum(axis=0) # q_batch_action [N, B]
                elif self.reweight_type == 'normalized_elu_linear':
                    assert not self.learnable_alpha, "normalized_elu_linear is not compatible with learnable_alpha"
                    assert self.alpha_transformation == 'identity', "normalized_elu_linear is not compatible with alpha_transformation != identity"
                    # q_min = get_min_q(next_obs, next_action)
                    batch_q_mean, batch_q_std = q_batch_action.mean(axis=0, keepdims=True), q_batch_action.std(axis=0, keepdims=True)
                    q_normalized = (q_batch_action  + alpha - batch_q_mean) / (batch_q_std + 1e-6)
                    q_weights = jax.nn.elu(q_normalized)
                    scaled_q = q_normalized
                    q_mean = batch_q_mean.mean()
                    q_std = batch_q_std.mean()
                    entropy_weights = jax.nn.relu(q_weights)
                    entropy = jax.scipy.special.entr(entropy_weights / entropy_weights.sum(axis=0, keepdims=True)).sum(axis=0) # q_batch_action [N, B]
                elif self.reweight_type == 'normalized_tanh_linear':
                    assert not self.learnable_alpha, "normalized_tanh_linear is not compatible with learnable_alpha"
                    assert self.alpha_transformation == 'identity', "normalized_tanh_linear is not compatible with alpha_transformation != identity"
                    # q_min = get_min_q(next_obs, next_action)
                    batch_q_mean, batch_q_std = q_batch_action.mean(axis=0, keepdims=True), q_batch_action.std(axis=0, keepdims=True)
                    q_normalized = (q_batch_action  + alpha - batch_q_mean) / (batch_q_std + 1e-6)
                    q_weights = jax.nn.tanh(q_normalized)
                    scaled_q = q_normalized
                    q_mean = batch_q_mean.mean()
                    q_std = batch_q_std.mean()
                    entropy_weights = jax.nn.relu(q_weights)
                    entropy = jax.scipy.special.entr(entropy_weights / entropy_weights.sum(axis=0, keepdims=True)).sum(axis=0) # q_batch_action [N, B]
                elif self.reweight_type == 'normalized_clipped_linear':
                    assert not self.learnable_alpha, "normalized_leaky_relu_linear is not compatible with learnable_alpha"
                    assert self.alpha_transformation == 'identity', "normalized_leaky_relu_linear is not compatible with alpha_transformation != identity"
                    # q_min = get_min_q(next_obs, next_action)
                    batch_q_mean, batch_q_std = q_batch_action.mean(axis=0, keepdims=True), q_batch_action.std(axis=0, keepdims=True)
                    q_normalized = (q_batch_action  + alpha - batch_q_mean) / (batch_q_std + 1e-6)
                    q_weights = jnp.clip(q_normalized, clipped_lower_bound, jnp.inf)
                    scaled_q = q_normalized
                    q_mean = batch_q_mean.mean()
                    q_std = batch_q_std.mean()
                    entropy_weights = jax.nn.relu(q_normalized)
                    entropy = jax.scipy.special.entr(entropy_weights / entropy_weights.sum(axis=0, keepdims=True)).sum(axis=0) # q_batch_action [N, B]
                elif self.reweight_type == 'normalized_clipped_square':
                    assert not self.learnable_alpha, "normalized_leaky_relu_linear is not compatible with learnable_alpha"
                    assert self.alpha_transformation == 'identity', "normalized_leaky_relu_linear is not compatible with alpha_transformation != identity"
                    # q_min = get_min_q(next_obs, next_action)
                    batch_q_mean, batch_q_std = q_batch_action.mean(axis=0, keepdims=True), q_batch_action.std(axis=0, keepdims=True)
                    q_normalized = (q_batch_action  + alpha - batch_q_mean) / (batch_q_std + 1e-6)
                    q_weights = jnp.clip(q_normalized, clipped_lower_bound, jnp.inf)
                    q_weights = jnp.where(q_weights > 0, q_weights ** 2, q_weights)
                    scaled_q = q_normalized
                    q_mean = batch_q_mean.mean()
                    q_std = batch_q_std.mean()
                    entropy_weights = jax.nn.relu(q_normalized)
                    entropy = jax.scipy.special.entr(entropy_weights / entropy_weights.sum(axis=0, keepdims=True)).sum(axis=0) # q_batch_action [N, B]
                elif self.reweight_type == 'logsumexp':
                    scaled_q = q_batch_action / alpha
                    Z = jax.nn.logsumexp(scaled_q, axis=0, keepdims=True)
                    q_weights = jnp.exp(scaled_q - Z)  # [N, B]
                    q_mean = jnp.mean(q_batch_action)
                    q_std = jnp.std(q_batch_action, axis=0).mean()
                    entropy = jax.scipy.special.entr(jax.nn.softmax(q_batch_action / alpha, axis=0)).sum(axis=0) # q_batch_action [N, B]
                elif self.reweight_type == 'strictly_normalized_logsumexp':
                    scaled_q = q_batch_action / alpha
                    Z = jax.nn.logsumexp(scaled_q, axis=0, keepdims=True)
                    q_weights = jnp.exp(scaled_q - Z) * self.agent.num_particles  # [N, B]
                    q_mean = jnp.mean(q_batch_action)
                    q_std = jnp.std(q_batch_action, axis=0).mean()
                    entropy = jax.scipy.special.entr(jax.nn.softmax(q_batch_action / alpha, axis=0)).sum(axis=0) # q_batch_action [N, B]
                elif self.reweight_type == 'negative_strictly_normalized_logsumexp':
                    scaled_q = q_batch_action / alpha
                    Z = jax.nn.logsumexp(scaled_q, axis=0, keepdims=True)
                    q_weights = jnp.exp(scaled_q - Z) * self.agent.num_particles + self.clipped_lower_bound  # [N, B]
                    q_mean = jnp.mean(q_batch_action)
                    q_std = jnp.std(q_batch_action, axis=0).mean()
                    entropy = jax.scipy.special.entr(jax.nn.softmax(q_batch_action / alpha, axis=0)).sum(axis=0) # q_batch_action [N, B]
                elif self.reweight_type == 'exp':
                    q_best_ind = jnp.argmax(q_batch_action, axis=0, keepdims=True)
                    act_best_of_n = jnp.take_along_axis(batch_action, q_best_ind[..., None], axis=0).squeeze(axis=0)
                    scaled_q = (q_batch_action - running_mean) / (running_std + 1e-6)
                    q_mean = jnp.mean(q_best_ind.squeeze(axis=0))
                    q_std = jnp.std(q_best_ind.squeeze(axis=0))
                else:
                    raise NotImplementedError(f"Reweight type {self.reweight_type} is not implemented.")

                if self.add_state_level_reweighting:
                    q_best_ind = jnp.argmax(q_batch_action, axis=0, keepdims=True) # [1, B]
                    q_rela = (q_best_ind - running_mean) / (running_std + 1e-6)
                    q_rela = q_rela.clip(-3, 3) / (jnp.exp(log_noise_scale) * 10.0) # fully reproduce dacer implementation
                    state_weights = jnp.exp(q_rela)
                    q_weights = q_weights * state_weights

                def denoiser(t, x):
                    return self.agent.policy(policy_params, obs, x, t)
                if self.agent.use_flow:
                    t = jax.random.uniform(diffusion_time_key, (self.agent.num_particles, obs.shape[0],))
                else:
                    t = jax.random.randint(diffusion_time_key, (self.agent.num_particles, obs.shape[0],), 0, self.agent.num_timesteps)

                # Bug #4 fix: Bind only keyword-only args in partial, pass all positional args via vmap
                loss_fn = partial(
                    self.agent.diffusion.weighted_p_loss,
                    negative_weights_regularization=self.negative_weights_regularization,
                    regularization_type=self.regularization_type,
                    clipped_only_weighted_mse_lower_bound=self.clipped_only_weighted_mse_lower_bound,
                    use_timestep_weight=self.use_timestep_weight)
                # vmap over: key (0), weights (0), model (None), t (0), x_start (0)
                # Each particle gets: unique key, unique weights vector, shared model, unique timesteps, unique actions
                loss, loss_info = jax.vmap(loss_fn, in_axes=(0, 0, None, 0, 0))(
                    diffusion_noise_keys,  # Shape (32,) -> each particle gets unique key
                    jax.lax.stop_gradient(q_weights),  # Shape (32, 256) -> each particle gets weights[i]
                    denoiser,  # Broadcast - same model for all particles
                    t,  # Shape (32, 256) -> each particle gets t[i]
                    jax.lax.stop_gradient(batch_action))  # Shape (32, 256, act_dim) -> each particle gets batch_action[i]
                loss_info = jax.tree.map(jnp.mean, loss_info)
                loss = jnp.mean(loss)
                return loss, (q_weights, scaled_q, q_mean, q_std, entropy, loss_info)

            (total_loss, (q_weights, scaled_q, q_mean, q_std, entropy, loss_info)), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)


            def alpha_loss_fn(alpha_variable: jax.Array) -> jax.Array:
                alpha = alpha_transform_fn(alpha_variable)
                alpha_loss = alpha * (self.kl_constraint + entropy.mean() - jnp.log(self.agent.num_particles))
                return alpha_loss.mean()

            alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(alpha_variable)


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

            policy_params, policy_opt_state = delay_param_update(self.policy_optim, policy_params, policy_grads, policy_opt_state)
            if self.learnable_alpha:
                alpha_variable, alpha_opt_state = param_update(self.alpha_optim, alpha_variable, alpha_grad, alpha_opt_state)
                if self.alpha_transformation == 'softplus':
                    alpha_variable = jnp.maximum(alpha_variable, softplus_inv(self.min_alpha))  # ensure alpha_variable is not too small
                elif self.alpha_transformation == 'exp':
                    alpha_variable = jnp.maximum(alpha_variable, jnp.log(self.min_alpha))  # ensure alpha_variable is not too small
                elif self.alpha_transformation == 'identity':
                    alpha_variable = jnp.maximum(alpha_variable, self.min_alpha)  # ensure alpha_variable is not too small
                else:
                    raise NotImplementedError(f"Alpha transformation {self.alpha_transformation} is not implemented.")
            else:
                pass

            if self.update_additive_noise_scale:
                def noise_scale_loss_fn(log_noise_scale: jax.Array) -> jax.Array:
                    return (jnp.exp(log_noise_scale) - self.target_noise_scale) ** 2

                noise_scale_loss, noise_scale_grad = jax.value_and_grad(noise_scale_loss_fn)(log_noise_scale)
                log_noise_scale, log_noise_scale_opt_state = jax.lax.cond(
                    step % self.delay_log_noise_scale_update == 0,
                    lambda params, opt_state: param_update(self.noise_optim, params, noise_scale_grad, opt_state),
                    lambda params, opt_state: (params, opt_state),
                    log_noise_scale, log_noise_scale_opt_state
                )

            else:
                log_noise_scale = jnp.log(self.target_noise_scale)
                log_noise_scale_opt_state = log_noise_scale_opt_state
                noise_scale_loss = 0.0


            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)
            target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)

            new_running_mean = running_mean + 0.001 * (q_mean - running_mean)
            new_running_std = running_std + 0.001 * (q_std - running_std)

            state = Diffv2TrainState(
                params=Diffv4Params(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, alpha_variable, log_noise_scale),
                opt_state=DPMDv2OptStates(
                    q1=q1_opt_state,
                    q2=q2_opt_state,
                    policy=policy_opt_state,
                    alpha_variable=alpha_opt_state,
                    log_noise_scale=log_noise_scale_opt_state
                    ),
                step=step + 1,
                entropy=jnp.float32(0.0),
                running_mean=new_running_mean,
                running_std=new_running_std
            )

            positive_q_weights_count = jnp.where(q_weights > 0, jnp.ones_like(q_weights), jnp.zeros_like(q_weights)).sum(axis=0)
            negative_q_weights_count = jnp.where(q_weights < 0, jnp.ones_like(q_weights), jnp.zeros_like(q_weights)).sum(axis=0)

            info = {
                "q1_loss": q1_loss,
                "q1_mean": jnp.mean(q1),
                "q1_max": jnp.max(q1),
                "q1_min": jnp.min(q1),
                "q2_loss": q2_loss,
                "td_error_abs_max": jnp.max(td_error_abs),
                "td_error_abs_mean": jnp.mean(td_error_abs),
                "td_error_abs_min": jnp.min(td_error_abs),
                "policy_loss": total_loss,
                "q_weights_std": jnp.std(q_weights),
                "q_weights_mean": jnp.mean(q_weights),
                "q_weights_min_min": jnp.min(q_weights),
                "q_weights_min_mean": jnp.min(q_weights, axis=0).mean(),
                "q_weights_max_mean": jnp.max(q_weights, axis=0).mean(),
                "q_weights_std_mean": jnp.std(q_weights, axis=0).mean(),
                "positive_q_weights_count_mean": jnp.mean(positive_q_weights_count),
                "positive_q_weights_count_std": jnp.std(positive_q_weights_count),
                "negative_q_weights_count_mean": jnp.mean(negative_q_weights_count),
                "negative_q_weights_count_std": jnp.std(negative_q_weights_count),
                "scale_q_mean": jnp.mean(scaled_q),
                "scale_q_std": jnp.std(scaled_q, axis=0).mean(),
                "scale_q_gap_mean": (jnp.max(scaled_q, axis=0) - jnp.min(scaled_q, axis=0)).mean(),
                "running_q_mean": new_running_mean,
                "running_q_std": new_running_std,
                # "approx_kl": jnp.mean(q_weights * (scaled_q - Z)),
                "kl_constraint": self.kl_constraint,
                "sample_action_std": jnp.std(batch_action, axis=0).mean(),

                # "normalization_factor": Z.squeeze(axis=0).mean(),
                "noise_scale_loss": noise_scale_loss,
                "noise_scale": jnp.exp(log_noise_scale),
                "alpha_variable": alpha_variable,
                "alpha": alpha,
                "alpha_loss": alpha_loss,
                "alpha_grad": alpha_grad,
                "analytical_alpha_grad": (self.kl_constraint + entropy - jnp.log(self.agent.num_particles)).mean(),
                "reweighted_entropy_mean": entropy.mean(),
                "reweighted_entropy_max": entropy.max(),
                "reweighted_entropy_min": entropy.min(),
                "reweighted_entropy_std": entropy.std(),
                "pos_loss": loss_info['pos_loss'],
                "neg_loss": loss_info['neg_loss'],
                "pos_unweighted_loss": loss_info['pos_mse'],
                "neg_unweighted_loss": loss_info['neg_mse'],
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)

    def get_policy_params(self):
        return (self.state.params.policy, self.state.params.log_noise_scale, self.state.params.q1, self.state.params.q2 )

    def get_policy_params_to_save(self):
        return (self.state.params.target_poicy, self.state.params.log_noise_scale, self.state.params.q1, self.state.params.q2)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)
