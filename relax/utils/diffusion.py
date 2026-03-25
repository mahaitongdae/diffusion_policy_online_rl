from typing import Protocol, Tuple
from dataclasses import dataclass

import numpy as np
import jax, jax.numpy as jnp
import optax

class DiffusionModel(Protocol):
    def __call__(self, t: jax.Array, x: jax.Array) -> jax.Array:
        ...


# class Diffusion:
    
#     @abstractmethod
#     def p_sample(self, key: jax.Array, model: DiffusionModel, shape: Tuple[int, ...]) -> jax.Array:
#         """Sample from the diffusion model."""
#         ...
    
#     @abstractmethod
#     def weighted_p_loss(self, key: jax.Array, weights: jax.Array, model: DiffusionModel, t: jax.Array,
#                         x_start: jax.Array) -> jax.Array:
#         """Compute the weighted loss for the diffusion model."""
#         ...

@dataclass(frozen=True)
class BetaScheduleCoefficients:
    betas: jax.Array
    alphas: jax.Array
    alphas_cumprod: jax.Array
    alphas_cumprod_prev: jax.Array
    sqrt_alphas_cumprod: jax.Array
    sqrt_one_minus_alphas_cumprod: jax.Array
    log_one_minus_alphas_cumprod: jax.Array
    sqrt_recip_alphas_cumprod: jax.Array
    sqrt_recipm1_alphas_cumprod: jax.Array
    posterior_variance: jax.Array
    posterior_log_variance_clipped: jax.Array
    posterior_mean_coef1: jax.Array
    posterior_mean_coef2: jax.Array

    @staticmethod
    def from_beta(betas: np.ndarray):
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = np.log(1. - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_log_variance_clipped = np.log(np.maximum(posterior_variance, 1e-20))
        posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)

        return BetaScheduleCoefficients(
            *jax.device_put((
                betas, alphas, alphas_cumprod, alphas_cumprod_prev,
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, log_one_minus_alphas_cumprod,
                sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod,
                posterior_variance, posterior_log_variance_clipped, posterior_mean_coef1, posterior_mean_coef2
            ))
        )

    @staticmethod
    def vp_beta_schedule(timesteps: int):
        t = np.arange(1, timesteps + 1)
        T = timesteps
        b_max = 10.
        b_min = 0.1
        alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
        betas = 1 - alpha
        return betas

    @staticmethod
    def cosine_beta_schedule(timesteps: int):
        s = 0.008
        t = np.arange(0, timesteps + 1) / timesteps
        alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        alphas_cumprod /= alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = np.clip(betas, 0, 0.999)
        return betas
    
    @staticmethod
    def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=0.999):
        return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)

@dataclass(frozen=True)
class GaussianDiffusion:
    num_timesteps: int
    beta_schedule_scale: float = 1.0
    beta_schedule_type: str = 'cosine'

    def beta_schedule(self):
        with jax.ensure_compile_time_eval():
            if self.beta_schedule_type == 'linear':
                betas = self.beta_schedule_scale * BetaScheduleCoefficients.linear_beta_schedule(self.num_timesteps)
            elif self.beta_schedule_type == 'cosine':
                betas = self.beta_schedule_scale * BetaScheduleCoefficients.cosine_beta_schedule(self.num_timesteps)
            return BetaScheduleCoefficients.from_beta(betas)

    def p_mean_variance(self, t: int, x: jax.Array, noise_pred: jax.Array):
        B = self.beta_schedule()
        x_recon = x * B.sqrt_recip_alphas_cumprod[t] - noise_pred * B.sqrt_recipm1_alphas_cumprod[t]
        x_recon = jnp.clip(x_recon, -1, 1)
        model_mean = x_recon * B.posterior_mean_coef1[t] + x * B.posterior_mean_coef2[t]
        model_log_variance = B.posterior_log_variance_clipped[t]
        return model_mean, model_log_variance
    
    def get_recon(self, t: int, x: jax.Array, noise: jax.Array):
        B = self.beta_schedule()
        x_recon = x * B.sqrt_recip_alphas_cumprod[t][:, jnp.newaxis] - noise * B.sqrt_recipm1_alphas_cumprod[t][:, jnp.newaxis]
        return x_recon

    def p_sample(self, key: jax.Array, model: DiffusionModel, shape: Tuple[int, ...]) -> jax.Array:
        B = self.beta_schedule()
        x_key, noise_key = jax.random.split(key)
        x = jnp.sqrt(1.0 - B.alphas_cumprod[-1]) * jax.random.normal(x_key, shape)
        noise = jax.random.normal(noise_key, (self.num_timesteps, *shape))

        def body_fn(x, input):
            t, noise = input
            noise_pred = model(t, x)
            model_mean, model_log_variance = self.p_mean_variance(t, x, noise_pred)
            x = model_mean + (t > 0) * jnp.exp(0.5 * model_log_variance) * noise
            return x, None

        t = jnp.arange(self.num_timesteps)[::-1]
        x, _ = jax.lax.scan(body_fn, x, (t, noise))
        return x

    def q_sample(self, t: int, x_start: jax.Array, noise: jax.Array):
        B = self.beta_schedule()
        return B.sqrt_alphas_cumprod[t] * x_start + B.sqrt_one_minus_alphas_cumprod[t] * noise

    def p_loss(self, key: jax.Array, model: DiffusionModel, t: jax.Array, x_start: jax.Array):
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]

        noise = jax.random.normal(key, x_start.shape)
        x_noisy = jax.vmap(self.q_sample)(t, x_start, noise)
        noise_pred = model(t, x_noisy)
        loss = optax.l2_loss(noise_pred, noise)
        return loss.mean()

    def weighted_p_loss(self, key: jax.Array, weights: jax.Array, model: DiffusionModel, t: jax.Array,
                        x_start: jax.Array, negative_weights_regularization: float = 0.0, regularization_type: 'str' = 'square',
                        clipped_only_weighted_mse_lower_bound: float = -1.0,
                        use_timestep_weight: bool = False):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        x_noisy = jax.vmap(self.q_sample)(t, x_start, noise)
        noise_pred = model(t, x_noisy)
        unweighted_loss = optax.squared_error(noise_pred, noise)
        if use_timestep_weight:
            B = self.beta_schedule()
            # Strict timestep weight from diffusion paper: \beta_t^2 / (2 \sigma_t^2 \alpha_t (1-\bar{\alpha}_t))
            # with \sigma_t^2 = posterior variance
            sigma_t_sq = B.posterior_variance[t]
            alpha_t = B.alphas[t]
            one_minus_bar_alpha_t = 1.0 - B.alphas_cumprod[t]
            timestep_weight = (B.betas[t] ** 2) / (2.0 * sigma_t_sq * alpha_t * one_minus_bar_alpha_t + 1e-12)
            unweighted_loss = unweighted_loss * timestep_weight.reshape(-1, 1)
        weights = weights * jnp.ones_like(unweighted_loss)
        if negative_weights_regularization > 0.0:
            if regularization_type == 'square':
                loss = unweighted_loss * weights
                loss += negative_weights_regularization * jnp.where(weights < 0, noise_pred ** 2, 0)
            elif regularization_type == 'softmax_with_mse':
                weighted_loss_mean_over_act = (weights * unweighted_loss).mean(axis=1, keepdims=True) # [N, 1]
                loss = jnp.where(
                    weights < 0, 
                    jax.nn.softmax(weighted_loss_mean_over_act, axis=0) * weighted_loss_mean_over_act, 
                    weighted_loss_mean_over_act
                    )
            elif regularization_type == 'softmax_with_mse_scaled':
                weighted_loss_mean_over_act = (weights * unweighted_loss).mean(axis=1, keepdims=True) # [N, 1]
                loss = jnp.where(
                    weights < 0, 
                    jax.nn.softmax(weighted_loss_mean_over_act, axis=0) * 32.0 * weighted_loss_mean_over_act, # TODO: hardcoded scale factor
                    weighted_loss_mean_over_act
                    )
            elif regularization_type == 'fpopp_like':
                # L = MSE capped at 1.0; negative samples: A*L + |A|/2 * L^2
                L = jnp.clip(unweighted_loss, 0.0, 1.0)
                loss = jnp.where(weights < 0, weights * L + jnp.abs(weights) / 2.0 * L ** 2, weights * unweighted_loss)
            elif regularization_type == 'fpopp_like_scaled':
                # L = MSE capped at 1.0; negative samples: A*L + |A|/2 * L^2
                L = jnp.clip(unweighted_loss, 0.0, 0.1)
                loss = jnp.where(weights < 0, weights * L + jnp.abs(weights) / 2.0 * L ** 2, weights * unweighted_loss)
            elif regularization_type == 'clipped_only':
                loss = jnp.where(weights < 0, jnp.clip(weights * unweighted_loss, clipped_only_weighted_mse_lower_bound, 0.0), weights * unweighted_loss)
            else:
                raise ValueError(f"Regularization type {regularization_type} is not implemented.")
        else:
            loss = weights * unweighted_loss
        pos_mask = (weights >= 0).any(axis=-1)
        neg_mask = (weights < 0).any(axis=-1)
        pos_loss = jnp.where(pos_mask, loss.mean(axis=-1), 0.0).sum() / jnp.maximum(pos_mask.sum(), 1)
        neg_loss = jnp.where(neg_mask, loss.mean(axis=-1), 0.0).sum() / jnp.maximum(neg_mask.sum(), 1)
        pos_mse = jnp.where(pos_mask, unweighted_loss.mean(axis=-1), 0.0).sum() / jnp.maximum(pos_mask.sum(), 1)
        neg_mse = jnp.where(neg_mask, unweighted_loss.mean(axis=-1), 0.0).sum() / jnp.maximum(neg_mask.sum(), 1)
        return loss.mean(), {'pos_loss': pos_loss, 'neg_loss': neg_loss, 'pos_mse': pos_mse, 'neg_mse': neg_mse}
    
    def reverse_samping_weighted_p_loss(self, noise: jax.Array, weights: jax.Array, model: DiffusionModel, t: jax.Array,
                        x_t: jax.Array):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_t.shape[0]
        noise_pred = model(t, x_t)
        loss = weights * optax.squared_error(noise_pred, noise)
        return loss.mean()
    

if __name__ == '__main__':
    diffusion = GaussianDiffusion(20)
    beta_schedule = diffusion.beta_schedule()
    print("betas", beta_schedule.betas)
    print("sqrt 1 - bar alpha", beta_schedule.sqrt_one_minus_alphas_cumprod)
    print("sqrt 1 over bar alpha", beta_schedule.sqrt_recip_alphas_cumprod)
    print("sqrt 1 - bar alpha over bar alpha", beta_schedule.sqrt_recipm1_alphas_cumprod)

