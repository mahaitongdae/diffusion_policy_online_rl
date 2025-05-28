from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle
import os
from pathlib import Path

from relax.algorithm.base import Algorithm
from relax.network.dacer import DACERNet, DACERParams
from relax.network.diffv2 import Diffv2Net, Diffv2Params
from relax.utils.experience import Experience
from relax.utils.typing_utils import Metric
from relax.utils.persistence import make_persist


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

class IDEM(Algorithm):

    def __init__(
        self,
        agent: Diffv2Net,
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

            # compute target q
            # next_action = self.agent.get_batch_actions(next_eval_key, (policy_params, log_alpha), next_obs, get_min_q)
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
            # q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            # q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            # q1_params = optax.apply_updates(q1_params, q1_update)
            # q2_params = optax.apply_updates(q2_params, q2_update)


            prev_entropy = state.entropy if hasattr(state, 'entropy') else jnp.float32(0.0)

            new_action = self.agent.get_action(new_eval_key, (policy_params, log_alpha, q1_params, q2_params), obs)
            diff_key1, diff_key2 = jax.random.split(diffusion_noise_key, 2)
            t = jax.random.randint(diffusion_time_key, (next_obs.shape[0],), 0, self.agent.num_timesteps)
            # lmbda = 0.8
            # steps = jnp.arange(0, self.agent.num_timesteps, dtype=jnp.float32)
            # weights_exponential = jnp.exp(-lmbda * steps + 1)
            # prob_exponential = weights_exponential / weights_exponential.sum()
            # t = jax.random.choice(diffusion_time_key, self.agent.num_timesteps, shape=(next_obs.shape[0],), p=prob_exponential)
            noise1 = jax.random.normal(diff_key1, action.shape)
            tilde_at = jax.vmap(self.agent.diffusion.q_sample)(t, new_action, noise1)
            # scale_ = self.agent.diffusion.beta_schedule().sqrt_alphas_cumprod[t][:, jnp.newaxis]
            # tilde_at = scale_ * new_action
            # tilde_at = jax.random.uniform(diff_key1, action.shape, minval=-1, maxval=1)
            # tilde_at = new_action
            
            
            
            def get_lse_by_idem(a) -> jax.Array:
                # Try muttiple samples to fit loss
                reverse_mc_num = 64
                wide_a = jnp.repeat(a, reverse_mc_num, axis=0)
                wide_t = jnp.repeat(t, reverse_mc_num, axis=0)
                wide_obs = jnp.repeat(obs, reverse_mc_num, axis=0)
                wide_noise = jax.random.normal(diff_key2, (action.shape[0] * reverse_mc_num, action.shape[1]))
                sample_a0 = self.agent.diffusion.get_recon(wide_t, wide_a, wide_noise).clip(-1, 1)
                q_min = get_min_q(wide_obs, sample_a0)
                # norm_q = (q_min - running_mean) / running_std # * 2. #  * 5. / jnp.exp(log_alpha)
                q_reshape = q_min.reshape([action.shape[0], reverse_mc_num, ])
                lse = jax.nn.logsumexp(q_reshape, axis=1)
                return lse.sum()
            noisy_score = jax.grad(get_lse_by_idem)(tilde_at)
            
            def policy_loss_fn(policy_params) -> jax.Array:
                def denoiser(t, x):
                    return self.agent.policy(policy_params, obs, x, t)
                noise_from_score = noisy_score * -1 * self.agent.diffusion.beta_schedule().sqrt_one_minus_alphas_cumprod[t][:, jnp.newaxis]
                return optax.squared_error(denoiser(t, tilde_at), noise_from_score).mean()
            

            total_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_params)

            # act_diff = jnp.linalg.norm(recon - action, axis=1)
            
            # update alpha
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                approx_entropy = 0.5 * self.agent.act_dim * jnp.log( 2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2)
                # log_alpha_loss = -jnp.mean(log_alpha * (-entropy + self.agent.target_entropy))
                log_alpha_loss = -1 * log_alpha * (-1 * jax.lax.stop_gradient(approx_entropy) + self.agent.target_entropy)
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
            log_alpha, log_alpha_opt_state = delay_alpha_param_update(self.alpha_optim, log_alpha, log_alpha_opt_state)

            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)
            target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)

            new_running_mean = running_mean # + 0.001 * (q_mean - running_mean)
            new_running_std = running_std # + 0.001 * (q_std - running_std)

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
                # "q1_std": jnp.mean(q1_std),
                "q2_loss": q2_loss,
                # "q2_mean": jnp.mean(q2_mean),
                # "q2_std": jnp.mean(q2_std),
                "policy_loss": total_loss,
                "alpha": jnp.exp(log_alpha),
                # "q_weights_std": jnp.std(q_weights),
                # "q_weights_mean": jnp.mean(q_weights),
                # "q_weights_min": jnp.min(q_weights),
                # "q_weights_max": jnp.max(q_weights),
                # "hist_q_weights": q_weights,
                "hist_t": t,
                # "scale_q_mean": jnp.mean(scaled_q),
                # "scale_q_std": jnp.std(scaled_q),
                "running_q_mean": new_running_mean,
                "running_q_std": new_running_std,
                # "act_diff_max": jnp.max(act_diff),
                # "act_diff_mean": jnp.mean(act_diff),
                # "act_diff_min": jnp.min(act_diff),
                # "mean_q1_std": mean_q1_std,
                # "mean_q2_std": mean_q2_std,
                # "entropy": entropy,
                "entropy_approx": 0.5 * self.agent.act_dim * jnp.log( 2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2),
            }
            return state, info

        self._implement_common_behavior(stateless_update, 
                                        self.agent.get_action, 
                                        self.agent.get_deterministic_action,
                                        stateless_get_value=self.agent.q)

    def get_policy_params(self):
        return (self.state.params.policy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2 )

    def get_policy_params_to_save(self):
        return (self.state.params.target_poicy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)
            
    def save_q_structure(self, root: os.PathLike, dummy_obs: jax.Array, dummy_action: jax.Array) -> None:
        root = Path(root)

        key = jax.random.key(0)
        deterministic = make_persist(self._get_value._fun)(self.get_value_params()[0], dummy_obs, dummy_action) # []

        deterministic.save(root / "q_func.pkl")
        deterministic.save_info(root / "q_func.txt")
            
    def get_value_params(self):
        return self.state.params.q1, self.state.params.q2
    
    def save_q(self, path: str) -> None:
        value = jax.device_get(self.get_value_params())
        with open(path, "wb") as f:
            pickle.dump(value, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)

def estimate_entropy(actions, num_components=3):  # (batch, sample, dim)
    import numpy as np
    from sklearn.mixture import GaussianMixture
    total_entropy = []
    for action in actions:
        gmm = GaussianMixture(n_components=num_components, covariance_type='full')
        gmm.fit(action)
        weights = gmm.weights_
        entropies = []
        for i in range(gmm.n_components):
            cov_matrix = gmm.covariances_[i]
            d = cov_matrix.shape[0]
            entropy = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * np.linalg.slogdet(cov_matrix)[1]
            entropies.append(entropy)
        entropy =  -np.sum(weights * np.log(weights)) + np.sum(weights * np.array(entropies))
        total_entropy.append(entropy)
    final_entropy = sum(total_entropy) / len(total_entropy)
    return final_entropy
