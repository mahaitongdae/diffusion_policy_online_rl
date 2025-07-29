import os
from dm_control import suite
import numpy as np

# Ensure offscreen rendering
os.environ["MUJOCO_GL"] = "osmesa"

# Load environment
env = suite.load(domain_name="cartpole", task_name="swingup")

# Take a step
time_step = env.reset()
action = np.random.uniform(env.action_spec().minimum, env.action_spec().maximum)
time_step = env.step(action)

print("Test successful! Observation received:", time_step.observation)
