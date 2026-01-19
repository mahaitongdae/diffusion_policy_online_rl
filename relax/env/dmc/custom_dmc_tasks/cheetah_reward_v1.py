# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Cheetah Domain."""

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.suite import cheetah as dmc_cheetah
from relax.env.dmc.custom_dmc_tasks import helper
SUITE = dmc_cheetah.SUITE

# How long the simulation will run, in seconds.
_DEFAULT_TIME_LIMIT = 10

# Reward levels after the _RUN_SPEED.
_RUN_SPEED = 10.0

# SUITE = containers.TaggedTasks()


import numpy as np

# def lqr_reward(speed):
#   return -(speed - _RUN_SPEED) ** 2 / 10

# def exp_lqr_reward(speed):
#   return np.exp(lqr_reward(speed))

# def linear_reward(speed):
#   return rewards.tolerance(speed,
#                                 bounds=(_RUN_SPEED, float('inf')),
#                                 margin=_RUN_SPEED / 2,
#                               value_at_margin=0.5,
#                                 sigmoid='linear',
#                                 )
# def exp_reward(speed):
#   return np.exp(linear_reward(speed))


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('cheetah.xml'), common.ASSETS
  
# @SUITE.add('benchmarking')
# def run_quadratic(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
#   """Returns the run task."""
#   physics = Physics.from_xml_string(*get_model_and_assets())
#   task = Cheetah(sigmoid='quadratic', random=random)
#   environment_kwargs = environment_kwargs or {}
#   return control.Environment(physics, task, time_limit=time_limit,
#                              **environment_kwargs)
  
# @SUITE.add('benchmarking')
# def run_reciprocal(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
#   """Returns the run task."""
#   physics = Physics.from_xml_string(*get_model_and_assets())
#   task = Cheetah(sigmoid='reciprocal', random=random)
#   environment_kwargs = environment_kwargs or {}
#   return control.Environment(physics, task, time_limit=time_limit,
#                              **environment_kwargs)
  
# @SUITE.add('benchmarking')
# def run_sparse(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
#   """Returns the run task."""
#   physics = Physics.from_xml_string(*get_model_and_assets())
#   task = Cheetah(sigmoid='sparse', random=random)
#   environment_kwargs = environment_kwargs or {}
#   return control.Environment(physics, task, time_limit=time_limit,
#                              **environment_kwargs)
  
@SUITE.add('benchmarking')
def run_lqr_v1(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cheetah(reward_type='lqr', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)
  
@SUITE.add('benchmarking')
def run_exp_lqr_v1(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cheetah(reward_type='exp_lqr', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)
@SUITE.add('benchmarking')
def run_exp_v1(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cheetah(reward_type='exp', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)
@SUITE.add('benchmarking')
def run_eval_v1(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cheetah(reward_type='eval', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)

@SUITE.add('benchmarking')
def run_square_v1(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cheetah(reward_type='square', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)

@SUITE.add('benchmarking')
def run_linear_v1(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cheetah(reward_type='linear', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)

@SUITE.add('benchmarking')
def run_abs_square_v1(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cheetah(reward_type='abs_square', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)

@SUITE.add('benchmarking')
def run_abs_sqrt_v1(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cheetah(reward_type='abs_sqrt', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)

@SUITE.add('benchmarking')
def run_abs_exp_v1(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cheetah(reward_type='abs_exp', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)

class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Cheetah domain."""

  def speed(self):
    """Returns the horizontal speed of the Cheetah."""
    return self.named.data.sensordata['torso_subtreelinvel'][0]


class Cheetah(base.Task):
  """A `Task` to train a running Cheetah."""
  
  def __init__(self, reward_type='linear', random=None):
    self._reward_type = reward_type
    super().__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""
    # The indexing below assumes that all joints have a single DOF.
    assert physics.model.nq == physics.model.njnt
    is_limited = physics.model.jnt_limited == 1
    lower, upper = physics.model.jnt_range[is_limited].T
    physics.data.qpos[is_limited] = self.random.uniform(lower, upper)

    # Stabilize the model before the actual simulation.
    physics.step(nstep=200)

    physics.data.time = 0
    self._timeout_progress = 0
    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of the state, ignoring horizontal position."""
    obs = collections.OrderedDict()
    # Ignores horizontal position to maintain translational invariance.
    obs['position'] = physics.data.qpos[1:].copy()
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""

    abs_reward = helper.AbsReward(5.0)
    if self._reward_type in ['lqr']:
      return helper.lqr_reward(physics.speed(), _RUN_SPEED)
    elif self._reward_type in ['linear']:
      return helper.linear_reward(physics.speed(), _RUN_SPEED)
    elif self._reward_type in ['exp_lqr']:
      return helper.exp_lqr_reward(physics.speed(), _RUN_SPEED)
    elif self._reward_type in ['exp']:
      return helper.exp_reward(physics.speed(), _RUN_SPEED)
    elif self._reward_type in ['square']:
      return helper.square_reward(physics.speed(), _RUN_SPEED)
    elif self._reward_type in ['abs_square']:
      return abs_reward.abs_square_reward(physics.speed())
    elif self._reward_type in ['abs_sqrt']:
      return abs_reward.abs_sqrt_reward(np.clip(physics.speed(), 1e-6, np.inf))
    elif self._reward_type in ['abs_exp']:
      return abs_reward.abs_exp_reward(physics.speed())
    elif self._reward_type in ['eval']:
      return physics.speed() / 10.0
    else:
      raise ValueError(f"Invalid reward type: {self._reward_type}")
