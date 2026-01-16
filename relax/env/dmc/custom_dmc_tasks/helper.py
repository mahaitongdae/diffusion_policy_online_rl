import numpy as np
from dm_control.utils import rewards


def lqr_reward(speed, speed_target):
  return -(np.clip(speed - speed_target, -np.inf, 0)) ** 2 / 10

def exp_lqr_reward(speed, speed_target):
  return np.exp(lqr_reward(speed, speed_target))

def linear_reward(speed, speed_target):
  return 2 * rewards.tolerance(speed,
                                bounds=(speed_target, float('inf')),
                                margin=speed_target / 2,
                              value_at_margin=0.5,
                                sigmoid='linear',
                                )
def exp_reward(speed, speed_target):
  return np.exp(1.15 * linear_reward(speed, speed_target))

def square_reward(speed, speed_target):
  return (linear_reward(speed, speed_target)) ** 2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    speed = np.linspace(0, 10, 100)

    speed_target = 8
    plt.figure(figsize=(5, 5))
    plt.plot(speed, lqr_reward(speed, speed_target), label='lqr')
    plt.plot(speed, exp_lqr_reward(speed, speed_target), label='exp_lqr')
    plt.plot(speed, linear_reward(speed, speed_target), label='linear')
    plt.plot(speed, exp_reward(speed, speed_target), label='exp')
    plt.plot(speed, square_reward(speed, speed_target), label='square')
    plt.legend()
    plt.savefig('reward.png')