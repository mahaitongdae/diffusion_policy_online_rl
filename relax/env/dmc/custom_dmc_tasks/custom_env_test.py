from dm_control import suite
# from relax.env.dmc.custom_dmc_tasks import cheetah_reward
from relax.env.dmc.custom_dmc_tasks import walker_reward

if __name__ == "__main__":
    env = suite.load('walker', 'walk_exp')