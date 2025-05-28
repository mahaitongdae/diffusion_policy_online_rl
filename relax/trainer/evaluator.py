import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import argparse
import pickle
import csv

import numpy as np
import jax
from tensorboardX import SummaryWriter

from relax.env import create_env
from relax.utils.persistence import PersistFunction

def evaluate(env, policy_fn, policy_params, num_episodes):
    ep_len_list = []
    ep_ret_list = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_len = 0
        ep_ret = 0.0
        while True:
            act = policy_fn(policy_params, obs)
            obs, reward, terminated, truncated, _ = env.step(act)
            ep_len += 1
            ep_ret += reward
            if terminated or truncated:
                break
        ep_len_list.append(ep_len)
        ep_ret_list.append(ep_ret)
    return ep_len_list, ep_ret_list

class Logger(object):

	def __init__(self, log_dir):
		self.path = os.path.join(log_dir, 'log.csv')
		with open(self.path, mode='w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(['step', 'avg_ret', 'std_ret'])

	def log(self, step, avg_ret, std_ret):
		with open(self.path, mode='a', newline='') as f:
			writer = csv.writer(f)
			writer.writerow([step, avg_ret, std_ret])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_root", type=Path)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    master_rng = np.random.default_rng(args.seed)
    env_seed, env_action_seed, policy_seed = map(int, master_rng.integers(0, 2**32 - 1, 3))
    env, _, _ = create_env(args.env, env_seed, env_action_seed)

    policy = PersistFunction.load(args.policy_root / "deterministic.pkl")
    @jax.jit
    def policy_fn(policy_params, obs):
        return policy(policy_params, obs).clip(-1, 1)

    # logger = SummaryWriter(args.policy_root)
    logger = Logger(args.policy_root)

    while payload := sys.stdin.readline():
        step, policy_path = payload.strip().split(",", maxsplit=1)
        step = int(step)
        with open(policy_path, "rb") as f:
            policy_params = pickle.load(f)

        ep_len_list, ep_ret_list = evaluate(env, policy_fn, policy_params, args.num_episodes)

        ep_len = np.array(ep_len_list)
        ep_ret = np.array(ep_ret_list)
        # logger.add_scalar("evaluate/episode_length", ep_len_mean.mean(), step)
        # logger.add_scalar("evaluate/episode_return", ep_ret_mean.mean(), step)
        # # logger.add_histogram("evaluate/episode_length", ep_len_mean, step)
        # # logger.add_histogram("evaluate/episode_return", ep_ret_mean, step)
        # logger.flush()
        logger.log(step, ep_ret.mean(), ep_ret.std())
