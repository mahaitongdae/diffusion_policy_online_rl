import math
import json
import subprocess
import sys
from pathlib import Path
from multiprocessing.shared_memory import SharedMemory

import gymnasium
from gymnasium.spaces import Box
import numpy as np

from relax.env.vector.base import VectorEnv

WORKER_PATH = Path(__file__).parent / "worker1.py"

class ProcessVectorEnv(VectorEnv):
    def __init__(self, name: str, num_envs: int, seed: int):
        self.num_envs = num_envs

        dummy_env = gymnasium.make(name)
        self.single_observation_space = dummy_env.observation_space
        self.single_action_space = dummy_env.action_space
        dummy_env.close()

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

        def make_shared_memory(shape, dtype):
            dtype = np.dtype(dtype)
            shm = SharedMemory(create=True, size=math.prod(shape) * dtype.itemsize)
            arr = np.ndarray(shape, dtype, shm.buf)
            descr = { "shape": shape, "dtype": dtype.str, "name": shm.name }
            return shm, arr, descr

        descr = {}
        self.obs_shm, self.obs, descr["obs"] = make_shared_memory((self.num_envs, self.obs_dim), np.float32)
        self.action_shm, self.action, descr["action"] = make_shared_memory((self.num_envs, self.act_dim), np.float32)
        self.reward_shm, self.reward, descr["reward"] = make_shared_memory((self.num_envs,), np.float64)
        self.terminated_shm, self.terminated, descr["terminated"] = make_shared_memory((self.num_envs,), np.bool_)
        self.truncated_shm, self.truncated, descr["truncated"] = make_shared_memory((self.num_envs,), np.bool_)

        def create_worker(i: int, seed: int):
            child = subprocess.Popen(
                [
                    sys.executable,
                    str(WORKER_PATH),
                    "--env", name,
                    "--index", str(i),
                    "--seed", str(seed),
                    "--descr", json.dumps(descr),
                ],
                bufsize=0,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            return child

        self.workers = [create_worker(i, seeds[i]) for i in range(num_envs)]

        self._shall_reset = np.ones((self.num_envs,), dtype=np.bool_)

    def reset(self, *, seed=None, options=None):
        assert seed is None and options is None

        for i, c in enumerate(self.workers):
            if self._shall_reset[i]:
                c.stdin.write(b"r")
        for i, c in enumerate(self.workers):
            if self._shall_reset[i]:
                assert c.stdout.read(1) == b"o"
        self._shall_reset[:] = False

        return self.obs.copy(), {}

    def step(self, action: np.ndarray):
        self.action[:] = action
        for c in self.workers:
            c.stdin.write(b"s")
        for c in self.workers:
            assert c.stdout.read(1) == b"o"
        self._shall_reset = self.terminated | self.truncated
        return self.obs.copy(), self.reward.copy(), self.terminated.copy(), self.truncated.copy(), {}

    def close(self):
        for c in self.workers:
            c.stdin.write(b"q")
        for c in self.workers:
            c.wait()
        for c in self.workers:
            c.stdout.close()
            c.stdin.close()
        for shm in [self.obs_shm, self.action_shm, self.reward_shm, self.terminated_shm, self.truncated_shm]:
            shm.close()
            shm.unlink()
        self.workers = None

    def __del__(self):
        if self.workers is not None:
            self.close()
