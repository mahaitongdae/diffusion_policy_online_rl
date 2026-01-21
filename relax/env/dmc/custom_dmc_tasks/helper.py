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

class AbsReward:
  def __init__(self, k_base, clip_input=None):
    from scipy.special import lambertw
    self.k_base = k_base
    self.k_sqrt = k_base * 1.0
    self.k_square = k_base * 1.25
    self.k_exp = k_base * 1.1
    self.alpha = float(lambertw(1 / k_base))
    self.sqrt_base = np.sqrt((k_base * k_base / 4 / self.k_sqrt) / self.k_sqrt)
    self.square_base = ((self.k_square * self.k_square / 2 / self.k_base) / self.k_square) ** 2
    self.exp_base = np.exp(self.alpha)

  def abs_square_reward(self, x):
    return ((x + self.k_square * self.k_square / 2 / self.k_base) / self.k_square) ** 2 - self.square_base

  def abs_sqrt_reward(self, x):
    return np.sqrt((x + self.k_base * self.k_base / 4 / self.k_sqrt) / self.k_sqrt) - self.sqrt_base

  def abs_exp_reward(self, x):
    return np.exp(x / self.k_exp + self.alpha) - self.exp_base 

  def abs_linear_reward(self, x):
    return x / self.k_base


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

    plt.figure(figsize=(5, 5))
    abs_reward = AbsReward(5.0)
    plt.plot(speed, abs_reward.abs_linear_reward(speed), label='abs_linear')
    plt.plot(speed, abs_reward.abs_square_reward(speed), label='abs_square')
    plt.plot(speed, abs_reward.abs_sqrt_reward(speed), label='abs_sqrt')
    plt.plot(speed, abs_reward.abs_exp_reward(speed), label='abs_exp')
    plt.legend()
    plt.savefig('abs_reward.png')

    plt.figure(figsize=(5, 5))
    abs_reward = AbsReward(0.5)
    clipped_speed = np.clip(speed, -np.inf, 1.0)
    plt.plot(speed, abs_reward.abs_linear_reward(clipped_speed), label='abs_linear')
    plt.plot(speed, abs_reward.abs_square_reward(clipped_speed), label='abs_square')
    plt.plot(speed, abs_reward.abs_sqrt_reward(clipped_speed), label='abs_sqrt')
    plt.plot(speed, abs_reward.abs_exp_reward(clipped_speed), label='abs_exp')
    plt.legend()
    plt.savefig('abs_reward_clipped.png')