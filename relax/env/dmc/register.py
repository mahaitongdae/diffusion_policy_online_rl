from turtle import onclick
from relax.env.dmc.wrapper import DMControlToGymWrapper
from relax.env.dmc.custom_dmc_tasks import cheetah_reward  # register cheetah_reward environment
from relax.env.dmc.custom_dmc_tasks import cheetah_reward_v1  # register cheetah_reward_v1 environment
from relax.env.dmc.custom_dmc_tasks import walker_reward_v1  # register walker_reward_v1 environment
from gymnasium.envs.registration import register
import gymnasium as gym
from dm_control import suite

def make_dm_control_env(domain_name, task_name, version=0, render_size=(640, 480), **kwargs):
    """Factory function to create a DMControlToGymWrapper environment."""
    if version == 1:
        task_name = task_name + '_v1'
    return DMControlToGymWrapper(domain_name, task_name, render_size=render_size, **kwargs)

# Register multiple DeepMind Control Suite environments
def register_dm_control_envs():
    custom_dm_control_envs = [
        ("cheetah", "run"),
        ("cheetah", "run_sparse"),
        ("cheetah", "run_quadratic"),
        ("cheetah", "run_reciprocal"),
        ("cheetah", "run_tanh_squared"),
        ("cheetah", "run_sparse_test"),
        ("cheetah", "run_lqr"),
        ("cheetah", "run_exp_lqr"),
        ("cheetah", "run_exp"),
        ("cheetah", "run_eval"),
        ("walker", "walk_lqr"),
        ("walker", "walk_exp_lqr"),
        ("walker", "walk_exp"),
        ("walker", "run_lqr"),
        ("walker", "run_exp_lqr"),
        ("walker", "run_exp"),
    ]
    custom_dm_control_envs_v1 = [
        ("cheetah", "run_lqr_v1"),
        ("cheetah", "run_exp_lqr_v1"),
        ("cheetah", "run_exp_v1"),
        ("cheetah", "run_eval_v1"),
        ("cheetah", "run_square_v1"),
        ("cheetah", "run_linear_v1"),
        ("cheetah", "run_abs_square_v1"),
        ("cheetah", "run_abs_sqrt_v1"),
        ("cheetah", "run_abs_exp_v1"),
        ("walker", "walk_lqr_v1"),
        ("walker", "walk_exp_lqr_v1"),
        ("walker", "walk_exp_v1"),
        ("walker", "walk_eval_v1"),
        ("walker", "walk_square_v1"),
        ("walker", "walk_linear_v1"),
        ("walker", "run_lqr_v1"),
        ("walker", "run_exp_lqr_v1"),
        ("walker", "run_exp_v1"),
        ("walker", "run_eval_v1"),
        ("walker", "run_square_v1"),
        ("walker", "run_linear_v1"),
        ("walker", "run_abs_square_v1"),
        ("walker", "run_abs_sqrt_v1"),
        ("walker", "run_abs_exp_v1"),
        ("walker", "run_abs_linear_v1"),
        ("walker", "walk_abs_square_v1"),
        ("walker", "walk_abs_sqrt_v1"),
        ("walker", "walk_abs_exp_v1"),
        ("walker", "walk_abs_linear_v1"),
    ]
    dm_control_envs = list(suite.ALL_TASKS)
    for env in custom_dm_control_envs:
        if env not in dm_control_envs:
            # print(f"env {env} not found in dm_control_envs")
            dm_control_envs.append(env)
    # print(dm_control_envs)
    for domain, task in dm_control_envs + custom_dm_control_envs_v1:
        if not task.endswith('_v1'):
            env_id = f"dm_control_{domain}_{task}-v0"
            version = 0
        else:
            task = task.replace('_v1', '')
            version = 1
            env_id = f"dm_control_{domain}_{task}-v1"
        if env_id not in gym.envs.registry.keys():
            register(
                id=env_id,
                entry_point=make_dm_control_env,
                kwargs={"domain_name": domain, "task_name": task, "version": version},
            )
        else:
            print(f"env {env_id} already registered, skipping")
        # print(domain, task)

"""
all avaliable envs:
Registered: dm_control_acrobot_swingup-v0
Registered: dm_control_acrobot_swingup_sparse-v0
Registered: dm_control_ball_in_cup_catch-v0
Registered: dm_control_cartpole_balance-v0
Registered: dm_control_cartpole_balance_sparse-v0
Registered: dm_control_cartpole_swingup-v0
Registered: dm_control_cartpole_swingup_sparse-v0
Registered: dm_control_cartpole_two_poles-v0
Registered: dm_control_cartpole_three_poles-v0
Registered: dm_control_cheetah_run-v0
Registered: dm_control_dog_stand-v0
Registered: dm_control_dog_walk-v0
Registered: dm_control_dog_trot-v0
Registered: dm_control_dog_run-v0
Registered: dm_control_dog_fetch-v0
Registered: dm_control_finger_spin-v0
Registered: dm_control_finger_turn_easy-v0
Registered: dm_control_finger_turn_hard-v0
Registered: dm_control_fish_upright-v0
Registered: dm_control_fish_swim-v0
Registered: dm_control_hopper_stand-v0
Registered: dm_control_hopper_hop-v0
Registered: dm_control_humanoid_stand-v0
Registered: dm_control_humanoid_walk-v0
Registered: dm_control_humanoid_run-v0
Registered: dm_control_humanoid_run_pure_state-v0
Registered: dm_control_humanoid_CMU_stand-v0
Registered: dm_control_humanoid_CMU_walk-v0
Registered: dm_control_humanoid_CMU_run-v0
Registered: dm_control_lqr_lqr_2_1-v0
Registered: dm_control_lqr_lqr_6_2-v0
Registered: dm_control_manipulator_bring_ball-v0
Registered: dm_control_manipulator_bring_peg-v0
Registered: dm_control_manipulator_insert_ball-v0
Registered: dm_control_manipulator_insert_peg-v0
Registered: dm_control_pendulum_swingup-v0
Registered: dm_control_point_mass_easy-v0
Registered: dm_control_point_mass_hard-v0
Registered: dm_control_quadruped_walk-v0
Registered: dm_control_quadruped_run-v0
Registered: dm_control_quadruped_escape-v0
Registered: dm_control_quadruped_fetch-v0
Registered: dm_control_reacher_easy-v0
Registered: dm_control_reacher_hard-v0
Registered: dm_control_stacker_stack_2-v0
Registered: dm_control_stacker_stack_4-v0
Registered: dm_control_swimmer_swimmer6-v0
Registered: dm_control_swimmer_swimmer15-v0
Registered: dm_control_walker_stand-v0
Registered: dm_control_walker_walk-v0
Registered: dm_control_walker_run-v0
"""

if __name__ == "__main__":
    import gymnasium as gym
    register_dm_control_envs()
    env = gym.make('dm_control_walker_run_lqr-v0')
    env_v1 = gym.make('dm_control_cheetah_run_lqr-v1')
    gym.make('dm_control_cheetah_run_linear-v1')
    gym.make('dm_control_cheetah_run_square-v1')
    gym.make('dm_control_cheetah_run_exp-v1')
    gym.make('dm_control_cheetah_run_exp_lqr-v1')
    gym.make('dm_control_walker_walk_lqr-v1')
    gym.make('dm_control_walker_walk_exp_lqr-v1')
    gym.make('dm_control_walker_walk_exp-v1')
    gym.make('dm_control_walker_walk_eval-v1')
    gym.make('dm_control_walker_walk_square-v1')
    gym.make('dm_control_walker_walk_linear-v1')
    gym.make('dm_control_walker_run_lqr-v1')
    gym.make('dm_control_walker_run_exp_lqr-v1')
    gym.make('dm_control_walker_run_exp-v1')
    gym.make('dm_control_walker_run_eval-v1')
    gym.make('dm_control_walker_run_square-v1')
    gym.make('dm_control_walker_run_linear-v1')
    gym.make('dm_control_cheetah_run_abs_square-v1')
    gym.make('dm_control_cheetah_run_abs_sqrt-v1')
    gym.make('dm_control_cheetah_run_abs_exp-v1')
    gym.make('dm_control_walker_run_abs_square-v1')
    gym.make('dm_control_walker_run_abs_sqrt-v1')
    gym.make('dm_control_walker_run_abs_exp-v1')
    gym.make('dm_control_walker_walk_abs_square-v1')
    gym.make('dm_control_walker_walk_abs_sqrt-v1')
    gym.make('dm_control_walker_walk_abs_exp-v1')
    gym.make('dm_control_walker_walk_abs_linear-v1')
    # env.reset()
    # for i in range(1000):
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     # print(obs, reward, terminated, truncated, info)
    #     if terminated or truncated:
    #         break
    # env.close()