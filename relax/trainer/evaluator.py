import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import argparse
import pickle
import csv
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import jax
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
		self._lock = threading.Lock()
		with open(self.path, mode='w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(['step', 'avg_ret', 'std_ret'])

	def log(self, step, avg_ret, std_ret):
		with self._lock:
			with open(self.path, mode='a', newline='') as f:
				writer = csv.writer(f)
				writer.writerow([step, avg_ret, std_ret])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_root", type=Path)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_eval_workers", type=int, default=4)
    args = parser.parse_args()

    master_rng = np.random.default_rng(args.seed)
    env_seed, env_action_seed, policy_seed = map(int, master_rng.integers(0, 2**32 - 1, 3))

    policy = PersistFunction.load(args.policy_root / "deterministic.pkl")
    @jax.jit
    def policy_fn(policy_params, obs):
        policy_output = policy(policy_params, obs)
        if isinstance(policy_output, tuple):
            act, _ = policy_output
        else:
            act = policy_output
        return act.clip(-1.0, 1.0)

    logger = Logger(args.policy_root)
    print_lock = threading.Lock()

    # Each worker thread gets its own env via thread-local storage
    _thread_local = threading.local()

    def get_thread_env():
        if not hasattr(_thread_local, "env"):
            # Each thread creates its own env with a unique seed
            seed_offset = threading.get_ident() % (2**31)
            _thread_local.env, _, _ = create_env(
                args.env,
                (env_seed + seed_offset) % (2**32),
                (env_action_seed + seed_offset) % (2**32),
            )
        return _thread_local.env

    def eval_worker(step, policy_params):
        env = get_thread_env()
        ep_len_list, ep_ret_list = evaluate(env, policy_fn, policy_params, args.num_episodes)

        ep_len = np.array(ep_len_list)
        ep_ret = np.array(ep_ret_list)

        logger.log(step, ep_ret.mean(), ep_ret.std())
        with print_lock:
            print(f"EVAL_METRICS:step={step},avg_ret={ep_ret.mean()},std_ret={ep_ret.std()},avg_len={ep_len.mean()}", flush=True)

    # Warm up JAX on the main thread env so jit compilation happens once
    warmup_env, _, _ = create_env(args.env, env_seed, env_action_seed)
    warmup_obs, _ = warmup_env.reset()
    dummy_params = None  # Will be populated on first eval
    warmup_env.close()

    with ThreadPoolExecutor(max_workers=args.num_eval_workers) as pool:
        while payload := sys.stdin.readline():
            step, policy_path = payload.strip().split(",", maxsplit=1)
            step = int(step)
            with open(policy_path, "rb") as f:
                policy_params = pickle.load(f)

            pool.submit(eval_worker, step, policy_params)
