from relax.env.dmc.wrapper import DMControlToGymWrapper
from gymnasium.envs.registration import register

def make_dm_control_env(domain_name, task_name, render_size=(640, 480), **kwargs):
    """Factory function to create a DMControlToGymWrapper environment."""
    return DMControlToGymWrapper(domain_name, task_name, render_size=render_size, **kwargs)

# Register multiple DeepMind Control Suite environments
def register_dm_control_envs():
    dm_control_envs = [
        ("quadruped", "walk"),
        ("quadruped", "run"),
        ("quadruped", "escape"),
        ("quadruped", "fetch"),
        ("walker", "stand"),
        ("walker", "run"),
        ("walker", "walk"),
        ("fish", "swim"),
        ("finger", "spin"),
        ("reacher", "hard"),
        ("cheetah", "run"),
        ("dog", "run"),
        ("humanoid", "run"),
    ]

    for domain, task in dm_control_envs:
        env_id = f"dm_control_{domain}_{task}-v0"
        register(
            id=env_id,
            entry_point=make_dm_control_env,
            kwargs={"domain_name": domain, "task_name": task},
        )
        print(domain, task)

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
