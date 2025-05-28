from typing import Optional
import gymnasium
from gymnasium.spaces import Box
from gymnasium.vector.async_vector_env import AsyncVectorEnv
import numpy as np

from relax.env.vector.base import VectorEnv


class ProcessVectorEnv(VectorEnv):
    def __init__(self, name: str, num_envs: int, seed: int):
        self.inner = AsyncVectorEnv([lambda: gymnasium.make(name) for _ in range(num_envs)])
        self.num_envs = num_envs

        self.single_observation_space = self.inner.single_observation_space
        self.single_action_space = self.inner.single_action_space

        assert isinstance(self.single_observation_space, Box) and len(self.single_observation_space.shape) == 1
        assert isinstance(self.single_action_space, Box) and len(self.single_action_space.shape) == 1 and self.single_action_space.is_bounded()

        self.obs_dim = self.single_observation_space.shape[0]
        self.act_dim = self.single_action_space.shape[0]

        def b(x):
            return np.broadcast_to(x, (self.num_envs, *x.shape))

        self.observation_space = Box(
            low=b(self.single_observation_space.low),
            high=b(self.single_observation_space.high),
            shape=(self.num_envs, self.obs_dim),
            dtype=self.single_observation_space.dtype,
        )

        self.action_space = Box(
            low=b(self.single_action_space.low),
            high=b(self.single_action_space.high),
            shape=(self.num_envs, self.act_dim),
            dtype=self.single_action_space.dtype,
        )

        if num_envs > 1:
            rng = np.random.default_rng(seed)
            seeds = rng.integers(0, 2**32 - 1, self.num_envs).tolist()
        else:
            seeds = [seed]
        self.inner.reset(seed=seeds)

        self.reset_obs = None

    def reset(self, *, seed=None, options=None):
        assert seed is None and options is None
        if self.reset_obs is not None:
            obs = self.reset_obs
            self.reset_obs = None
        else:
            print("Force reset")
            obs, _ = self.inner.reset()
        return obs, {}

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.inner.step(action)
        if "final_observation" in info:
            self.reset_obs = obs.copy()
            final_observation = info["final_observation"]
            for i in range(self.num_envs):
                if final_observation[i] is not None:
                    obs[i] = final_observation[i]
        return obs, reward, terminated, truncated, info

    def close(self):
        self.inner.close()
