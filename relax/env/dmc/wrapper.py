import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import suite
from dm_env import StepType

class DMControlToGymWrapper(gym.Env):
    """Wrapper to convert DeepMind Control Suite env to a Gymnasium-compatible environment."""

    def __init__(self, domain_name, task_name, env=None, render_size=(640, 480)):
        super().__init__()

        # Load the DeepMind Control Suite environment
        if env is not None:
            self.env = env
        else:
            self.env = suite.load(domain_name=domain_name, task_name=task_name)
        self.render_size = render_size

        # Extract action and observation space
        self.action_spec = self.env.action_spec()
        self.observation_spec = self.env.observation_spec()

        # Define Gym action space (continuous)
        self.action_space = spaces.Box(
            low=self.action_spec.minimum.astype(np.float32),
            high=self.action_spec.maximum.astype(np.float32),
            dtype=np.float32
        )

        # Flatten observation space and define it in Gym
        obs_dim = sum(np.prod(spec.shape) for spec in self.observation_spec.values()).astype(int)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _flatten_observation(self, time_step):
        """Flattens the observation dictionary into a single numpy array."""
        return np.concatenate([np.asarray(time_step.observation[key]).flatten() for key in self.observation_spec])

    def reset(self, seed=None, options=None):
        """Resets the environment and returns the initial observation."""
        if seed is not None:
            np.random.seed(seed)
        time_step = self.env.reset()
        return self._flatten_observation(time_step).astype(np.float32), {}

    def step(self, action):
        """Steps through the environment."""
        time_step = self.env.step(action)

        obs = self._flatten_observation(time_step).astype(np.float32)
        reward = time_step.reward if time_step.reward is not None else 0.0
        terminated = time_step.step_type == StepType.LAST
        truncated = False  # DMC does not define truncation explicitly

        return obs, reward, terminated, truncated, {}

    def render(self):
        """Renders the environment as an image."""
        return self.env.physics.render(*self.render_size, camera_id=0)

    def close(self):
        """Closes the environment."""
        pass

# Example usage:
if __name__ == "__main__":
    env = DMControlToGymWrapper("quadruped", "walk")

    obs, info = env.reset()
    print("Initial observation:", obs)

    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print("Step observation:", obs, "Reward:", reward, "Done:", done)

    env.close()
