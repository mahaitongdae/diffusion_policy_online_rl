from pathlib import Path
from typing import Callable, Optional, Tuple

import jax
from numba import njit, types as nt
import numpy as np
from gymnasium import Env
from tqdm import tqdm
from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams

from relax.algorithm import Algorithm
from relax.env import RelaxWrapper
from relax.env.vector import VectorEnv
from relax.trainer.accumulator import Interval, VectorFragmentSampleLog
from relax.utils.experience import GAEExperience

class OnPolicySampler:
    def __init__(
        self,
        env: Env,
        algorithm: Algorithm,
        batch_size: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        assert isinstance(env.unwrapped, VectorEnv) and isinstance(env, RelaxWrapper)
        assert batch_size % env.unwrapped.num_envs == 0

        self.env = env
        self.algorithm = algorithm
        self.rollout_fragment_length = batch_size // env.unwrapped.num_envs
        self.num_envs = env.unwrapped.num_envs

        self.obs_buf = np.zeros((self.num_envs, self.rollout_fragment_length, self.env.obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((self.num_envs, self.rollout_fragment_length, self.env.act_dim), dtype=np.float32)
        self.reward_buf = np.zeros((self.num_envs, self.rollout_fragment_length), dtype=np.float64)
        self.terminated_buf = np.zeros((self.num_envs, self.rollout_fragment_length), dtype=np.bool_)
        self.truncated_buf = np.zeros((self.num_envs, self.rollout_fragment_length), dtype=np.bool_)
        self.next_obs_buf = np.zeros((self.num_envs, self.rollout_fragment_length, self.env.obs_dim), dtype=np.float32)

        self.obs, _ = self.env.reset()
        self.log = VectorFragmentSampleLog(self.num_envs, self.rollout_fragment_length)

        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def sample(self, keys: jax.Array):
        for i in range(self.rollout_fragment_length):
            action = self.algorithm.get_action(keys[i], self.obs)
            action_clipped = np.clip(action, -1, 1)  # Crucial!
            next_obs, reward, terminated, truncated, _ = self.env.step(action_clipped)
            any_done = np.any(terminated) or np.any(truncated)
            self.obs_buf[:, i] = self.obs
            self.action_buf[:, i] = action
            self.reward_buf[:, i] = reward
            self.terminated_buf[:, i] = terminated
            self.truncated_buf[:, i] = truncated
            self.next_obs_buf[:, i] = next_obs
            if any_done:
                self.obs, _ = self.env.reset()
            else:
                self.obs = next_obs

        self.log.add(self.reward_buf, self.terminated_buf, self.truncated_buf, {})

        self.truncated_buf[:, -1] = True
        self.truncated_buf &= ~self.terminated_buf

        obs_buf = self.obs_buf.reshape(-1, self.env.obs_dim)
        action_buf = self.action_buf.reshape(-1, self.env.act_dim)
        reward_buf = self.reward_buf.reshape(-1)
        terminated_buf = self.terminated_buf.reshape(-1)
        truncated_buf = self.truncated_buf.reshape(-1)
        next_obs_buf = self.next_obs_buf.reshape(-1, self.env.obs_dim)

        # Compute truncated value
        value_buf = self.algorithm.get_value(obs_buf)
        truncated_pos, truncated_value_buf = self.compute_truncated_value(next_obs_buf, truncated_buf, self.algorithm.get_value)
        terminated_pos, = terminated_buf.nonzero()

        # Compute advantage and return
        adv_buf, ret_buf = compute_gae(reward_buf, value_buf, terminated_pos, truncated_pos, truncated_value_buf, self.gamma, self.gae_lambda)

        return GAEExperience(obs_buf, action_buf, reward_buf, terminated_buf, next_obs_buf, ret_buf, adv_buf)

    def compute_truncated_value(self, next_obs_buf: np.ndarray, truncated_buf: np.ndarray, value_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        truncated_pos, = truncated_buf.nonzero()
        truncated_size = len(truncated_pos)
        if truncated_size == 0:
            return truncated_pos, np.zeros((0,), dtype=np.float32)
        truncated_pad_size = 2 ** (truncated_size - 1).bit_length()
        if truncated_pad_size == truncated_size:
            return truncated_pos, value_fn(next_obs_buf[truncated_pos])
        else:
            truncated_next_obs_buf = np.empty((truncated_pad_size, *next_obs_buf.shape[1:]), dtype=np.float32)
            np.take(next_obs_buf, truncated_pos, axis=0, out=truncated_next_obs_buf[:truncated_size])
            truncated_value = value_fn(truncated_next_obs_buf)
            return truncated_pos, truncated_value[:truncated_size].copy()

@njit([(nt.float64[::1], nt.Array(nt.float32, 1, 'C', readonly=True), nt.int64[::1], nt.int64[::1], nt.Array(nt.float32, 1, 'C', readonly=True), nt.float64, nt.float64)], cache=True, fastmath=True)
def compute_gae(reward: np.ndarray, value: np.ndarray, terminated_pos: np.ndarray, truncated_pos: np.ndarray, truncated_value: np.ndarray, gamma: float, gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
    def compute_fragment_gae(reward: np.ndarray, value: np.ndarray, adv: np.ndarray, ret: np.ndarray, left: int, right: int, truncated_value: float, gamma: float, gae_lambda: float):
        ret[right - 1] = final_ret = reward[right - 1] + gamma * truncated_value
        adv[right - 1] = gae = final_ret - value[right - 1]
        for i in range(right - 2, left - 1, -1):
            delta = reward[i] + gamma * value[i + 1] - value[i]
            gae = delta + gamma * gae_lambda * gae
            ret[i] = gae + value[i]
            adv[i] = gae

    batch_size = len(reward)
    adv_buf = np.empty((batch_size,), dtype=np.float32)
    ret_buf = np.empty((batch_size,), dtype=np.float32)
    left = 0
    terminated_ptr = 0
    truncated_ptr = 0
    num_terminated = len(terminated_pos)
    num_truncated = len(truncated_pos)
    while terminated_ptr < num_terminated and truncated_ptr < num_truncated:
        terminated_idx = terminated_pos[terminated_ptr]
        truncated_idx = truncated_pos[truncated_ptr]
        if terminated_idx < truncated_idx:
            right = terminated_idx + 1
            compute_fragment_gae(reward, value, adv_buf, ret_buf, left, right, 0.0, gamma, gae_lambda)
            left = right
            terminated_ptr += 1
        else:
            right = truncated_idx + 1
            compute_fragment_gae(reward, value, adv_buf, ret_buf, left, right, truncated_value[truncated_ptr], gamma, gae_lambda)
            left = right
            truncated_ptr += 1

    while terminated_ptr < num_terminated:
        terminated_idx = terminated_pos[terminated_ptr]
        right = terminated_idx + 1
        compute_fragment_gae(reward, value, adv_buf, ret_buf, left, right, 0.0, gamma, gae_lambda)
        left = right
        terminated_ptr += 1

    while truncated_ptr < num_truncated:
        truncated_idx = truncated_pos[truncated_ptr]
        right = truncated_idx + 1
        compute_fragment_gae(reward, value, adv_buf, ret_buf, left, right, truncated_value[truncated_ptr], gamma, gae_lambda)
        left = right
        truncated_ptr += 1

    assert left == batch_size  # Either terminated or truncated should be the last element

    return adv_buf, ret_buf

class OnPolicyTrainer:
    def __init__(
        self,
        env: Env,
        algorithm: Algorithm,
        log_path: Path,
        batch_size: int = 4000,
        total_step: int = int(1e7),
        sample_log_n_episode: int = 10,
        save_policy_every: int = 10000,
        hparams: Optional[dict] = None,
        policy_pkl_template: str = "policy-{sample_step}.pkl",
    ):
        self.algorithm = algorithm
        self.batch_size = batch_size
        self.total_step = total_step
        self.log_path = log_path
        self.policy_pkl_template = policy_pkl_template
        self.sample_log_n_episode = sample_log_n_episode
        self.save_policy_every = save_policy_every
        self.hparams = hparams
        # TODO: make EpisodeLog and Experience configurable
        # TODO: re-add done_info_keys support
        # TODO: re-add evaluation support

        assert total_step % batch_size == 0
        self.total_iter = total_step // batch_size

        self.sampler = OnPolicySampler(env, algorithm, self.batch_size, gamma=algorithm.gamma, gae_lambda=algorithm.gae_lambda)
        self.sample_log_interval = Interval(self.sample_log_n_episode)
        self.last_metrics = {}

    def setup(self, dummy_data: GAEExperience):
        self.algorithm.warmup(dummy_data)

        # Setup logger
        self.logger = SummaryWriter(str(self.log_path))
        self.progress = tqdm(range(self.total_iter), desc="Train Step", disable=None, dynamic_ncols=True)

        self.algorithm.save_policy_structure(self.log_path, dummy_data.obs[0])

    def train(self, key: jax.Array):
        iter_key_fn = create_iter_key_fn(key, self.sampler.rollout_fragment_length)

        for i in self.progress:
            sample_keys, update_key = iter_key_fn(i)

            experience = self.sampler.sample(sample_keys)
            if self.sample_log_interval.check(self.sampler.log.sample_episode):
                self.sampler.log.log(self.add_scalar)

            metrics = self.algorithm.update(update_key, experience)
            for k, v in metrics.items():
                self.add_scalar(f"update/{k}", v, i * self.batch_size)

            if i % self.save_policy_every == 0:
                policy_pkl_name = self.policy_pkl_template.format(
                    sample_step=i * self.batch_size,
                )
                self.algorithm.save_policy(self.log_path / policy_pkl_name)

    def add_scalar(self, tag: str, value: float, step: int):
        self.last_metrics[tag] = value
        self.logger.add_scalar(tag, value, step)

    def run(self, key: jax.Array):
        try:
            self.train(key)
        except KeyboardInterrupt:
            pass
        finally:
            self.finish()

    def finish(self):
        self.sampler.env.close()
        self.algorithm.save(self.log_path / "state.pkl")
        if self.hparams is not None and len(self.last_metrics) > 0:
            exp, ssi, sei = hparams(self.hparams, self.last_metrics)
            self.logger.file_writer.add_summary(exp)
            self.logger.file_writer.add_summary(ssi)
            self.logger.file_writer.add_summary(sei)
        self.logger.close()
        self.progress.close()

def create_iter_key_fn(key: jax.Array, rollout_fragment_length: int) -> Callable[[int], Tuple[jax.Array, jax.Array]]:
    def iter_key_fn(step: int):
        iter_key = jax.random.fold_in(key, step)
        sample_key, update_key = jax.random.split(iter_key)
        sample_keys = jax.random.split(sample_key, rollout_fragment_length)
        return sample_keys, update_key

    iter_key_fn = jax.jit(iter_key_fn)
    iter_key_fn(0)  # Warm up
    return iter_key_fn
