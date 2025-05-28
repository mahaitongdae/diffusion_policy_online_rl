from typing import Optional
import gymnasium
from gymnasium.spaces import Box
import numpy as np

from relax.env.vector.base import VectorEnv


class SerialVectorEnv(VectorEnv):
    def __init__(self, name: str, num_envs: int, seed: int):
        assert num_envs > 0

        self.num_envs = num_envs
        self.envs = [gymnasium.make(name) for _ in range(num_envs)]

        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

        assert isinstance(self.single_observation_space, Box) and len(self.single_observation_space.shape) == 1
        assert isinstance(self.single_action_space, Box) and len(self.single_action_space.shape) == 1 and self.single_action_space.is_bounded()
        for env in self.envs:
            assert env.observation_space == self.single_observation_space
            assert env.action_space == self.single_action_space

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

        self._observation = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        self._reward = np.zeros((self.num_envs,), dtype=np.float64)
        self._termination = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncation = np.zeros((self.num_envs,), dtype=np.bool_)

        if num_envs > 1:
            rng = np.random.default_rng(seed)
            seeds = rng.integers(0, 2**32 - 1, self.num_envs).tolist()
        else:
            seeds = [seed]
        for i in range(self.num_envs):
            self.envs[i].reset(seed=seeds[i])

        self._shall_reset = np.ones((self.num_envs,), dtype=np.bool_)

    def reset(self, *, seed=None, options=None):
        assert seed is None and options is None
        for i in range(self.num_envs):
            if self._shall_reset[i]:
                self._observation[i], _ = self.envs[i].reset()
        self._shall_reset[:] = False

        return self._observation.copy(), {}

    def step(self, action: np.ndarray):
        assert action.shape == (self.num_envs, self.act_dim)

        for i in range(self.num_envs):
            self._observation[i], self._reward[i], self._termination[i], self._truncation[i], _ = self.envs[i].step(action[i])
        self._shall_reset = self._termination | self._truncation

        return self._observation.copy(), self._reward.copy(), self._termination.copy(), self._truncation.copy(), {}

    def close(self):
        for env in self.envs:
            env.close()

    @property
    def unwrapped(self):
        return self
