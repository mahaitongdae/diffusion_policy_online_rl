import os
from pathlib import Path
import pickle

import numpy as np
import jax, jax.numpy as jnp

from relax.utils.experience import Experience
from relax.utils.persistence import make_persist
from relax.utils.typing_utils import Metric


class Algorithm:
    # NOTE: a not elegant blanket implementation of the algorithm interface
    def _implement_common_behavior(self, stateless_update, stateless_get_action, stateless_get_deterministic_action, stateless_get_value=None):
        self._update = jax.jit(stateless_update)
        self._get_action = jax.jit(stateless_get_action)
        self._get_deterministic_action = jax.jit(stateless_get_deterministic_action)
        if stateless_get_value is not None:
            self._get_value = jax.jit(stateless_get_value)

    def update(self, key: jax.Array, data: Experience) -> Metric:
        self.state, info = self._update(key, self.state, data)
        return {k: float(v) for k, v in info.items() if not k.startswith('hist')}, {k: v for k, v in info.items() if k.startswith('hist')}

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params(), obs)
        return np.asarray(action)

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        action = self._get_deterministic_action(self.get_policy_params(), obs)
        return np.asarray(action)

    def get_value(self, obs: np.ndarray) -> np.ndarray:
        value = self._get_value(self.get_value_params(), obs)
        return np.asarray(value)

    def save(self, path: str) -> None:
        state = jax.device_get(self.state)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.state = jax.device_put(state)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params())
        with open(path, "wb") as f:
            pickle.dump(policy, f)
            
    def save_q(self, path: str) -> None:
        policy = jax.device_get(self.get_value_params())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def save_policy_structure(self, root: os.PathLike, dummy_obs: jax.Array) -> None:
        root = Path(root)

        key = jax.random.key(0)
        stochastic = make_persist(self._get_action._fun)(key, self.get_policy_params(), dummy_obs)
        deterministic = make_persist(self._get_deterministic_action._fun)(self.get_policy_params(), dummy_obs)

        stochastic.save(root / "stochastic.pkl")
        stochastic.save_info(root / "stochastic.txt")
        deterministic.save(root / "deterministic.pkl")
        deterministic.save_info(root / "deterministic.txt")

    def save_q_structure(self, root: os.PathLike, dummy_obs: jax.Array, dummy_action: jax.Array) -> None:
        root = Path(root)

        key = jax.random.key(0)
        # stochastic = make_persist(self._get_value._fun)(key, self.get_policy_params(), dummy_obs)
        deterministic = make_persist(self._get_value._fun)(self.get_value_params(), dummy_obs, dummy_action) # []

        # stochastic.save(root / "stochastic.pkl")
        # stochastic.save_info(root / "stochastic.txt")
        deterministic.save(root / "q_func.pkl")
        deterministic.save_info(root / "q_func.txt")

    def get_policy_params(self):
        return self.state.params.policy

    def get_value_params(self):
        return self.state.params.value

    def warmup(self, data: Experience) -> None:
        key = jax.random.key(0)
        obs = data.obs[0]
        policy_params = self.get_policy_params()
        self._update(key, self.state, data)
        self._get_action(key, policy_params, obs)
        self._get_deterministic_action(policy_params, obs)
