from typing import NamedTuple, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax

def probe_batch_size(reward: "jax.Array") -> Optional[int]:
    try:
        if reward.ndim > 0:
            return reward.shape[0]
        else:
            return None
    except AttributeError:
        return None

class Experience(NamedTuple):
    obs: "jax.Array"
    action: "jax.Array"
    reward: "jax.Array"
    done: "jax.Array"
    next_obs: "jax.Array"

    def batch_size(self) -> Optional[int]:
        return probe_batch_size(self.reward)

    def __repr__(self):
        return f"Experience(size={self.batch_size()})"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return Experience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=np.zeros(leading_dims, dtype=np.float32),
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            done=np.zeros(leading_dims, dtype=np.bool_),
        )

    @staticmethod
    def create(obs, action, reward, terminated, truncated, next_obs, info=None):
        return Experience(obs=obs, action=action, reward=reward, done=terminated, next_obs=next_obs)

class GAEExperience(NamedTuple):
    obs: "jax.Array"
    action: "jax.Array"
    reward: "jax.Array"
    done: "jax.Array"
    next_obs: "jax.Array"
    ret: "jax.Array"
    adv: "jax.Array"

    def batch_size(self) -> Optional[int]:
        return probe_batch_size(self.reward)

    def __repr__(self):
        return f"GAEExperience(size={self.batch_size()})"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return GAEExperience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=np.zeros(leading_dims, dtype=np.float32),
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            done=np.zeros(leading_dims, dtype=np.bool_),
            ret=np.zeros(leading_dims, dtype=np.float32),
            adv=np.zeros(leading_dims, dtype=np.float32),
        )

class SafeExperience(NamedTuple):
    obs: "jax.Array"
    action: "jax.Array"
    reward: "jax.Array"
    done: "jax.Array"
    next_obs: "jax.Array"
    cost: "jax.Array"
    feasible: "jax.Array"
    infeasible: "jax.Array"
    barrier: "jax.Array"
    next_barrier: "jax.Array"

    def batch_size(self) -> Optional[int]:
        try:
            if self.reward.ndim > 0:
                return self.reward.shape[0]
            else:
                return None
        except AttributeError:
            return None

    def __repr__(self):
        return f"SafeExperience(size={self.batch_size()})"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return SafeExperience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=np.zeros(leading_dims, dtype=np.float32),
            done=np.zeros(leading_dims, dtype=np.bool_),
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            cost=np.zeros(leading_dims, dtype=np.float32),
            feasible=np.zeros(leading_dims, dtype=np.bool_),
            infeasible=np.zeros(leading_dims, dtype=np.bool_),
            barrier=np.zeros(leading_dims, dtype=np.float32),
            next_barrier=np.zeros(leading_dims, dtype=np.float32),
        )

    @staticmethod
    def create(obs, action, reward, terminated, truncated, next_obs, info: dict):
        cost = info.get("cost", 0.0)
        feasible = info.get("feasible", False)
        infeasible = info.get("infeasible", False)
        barrier = info.get("barrier", 0.0)
        next_barrier = info.get("next_barrier", 0.0)
        return SafeExperience(
            obs=obs,
            action=action,
            reward=reward,
            done=terminated,
            next_obs=next_obs,
            cost=cost,
            feasible=feasible,
            infeasible=infeasible,
            barrier=barrier,
            next_barrier=next_barrier,
        )

class ObsActionPair(NamedTuple):
    obs: "jax.Array"
    action: "jax.Array"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return ObsActionPair(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
        )
