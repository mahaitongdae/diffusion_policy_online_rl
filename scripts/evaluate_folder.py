#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import argparse
import pickle
import csv
import glob
from datetime import datetime
import yaml
import numpy as np

# Set environment variables for JAX before importing it
# os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"

# Add project root to sys.path to allow imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    import jax
    from relax.env import create_env
    from relax.utils.persistence import PersistFunction
    from relax.trainer.evaluator import evaluate
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure you are running this from the project root or have the project in your PYTHONPATH.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Evaluate all saved policies in a log directory.")
    parser.add_argument("log_dir", type=Path, help="Path to the directory containing policy-*.pkl files and config.yaml")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate for each policy")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for evaluation")
    parser.add_argument("--eval_env", type=str, default="None", help="Environment to evaluate in")
    args = parser.parse_args()

    log_dir = args.log_dir.resolve()
    if not log_dir.exists():
        print(f"Error: Directory {log_dir} does not exist.")
        sys.exit(1)

    config_path = log_dir / "config.yaml"
    if not config_path.exists():
        print(f"Error: config.yaml not found in {log_dir}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Use eval_env if specified, otherwise fall back to env
    env_id = args.eval_env if args.eval_env != "None" else config.get("eval_env", config.get("env"))
    print(f"Evaluating in environment: {env_id}")
    
    master_rng = np.random.default_rng(args.seed)
    env_seed, env_action_seed, _ = map(int, master_rng.integers(0, 2**32 - 1, 3))
    env, _, _ = create_env(env_id, env_seed, env_action_seed)

    # Load base policy
    deterministic_pkl = log_dir / "deterministic.pkl"
    if not deterministic_pkl.exists():
        print(f"Error: {deterministic_pkl} not found. Needed for policy structure.")
        sys.exit(1)
        
    policy = PersistFunction.load(deterministic_pkl)

    @jax.jit
    def policy_fn(policy_params, obs):
        policy_output = policy(policy_params, obs)
        if isinstance(policy_output, tuple):
            act, _ = policy_output
        else:
            act = policy_output
        return act.clip(-1.0, 1.0)

    # Find all policies
    policy_files = glob.glob(str(log_dir / "policy-*.pkl"))
    
    def get_step(fpath):
        # Extract sample_step from policy-{sample_step}-{update_step}.pkl
        name = os.path.basename(fpath)
        parts = name.replace(".pkl", "").split("-")
        try:
            return int(parts[1])
        except (IndexError, ValueError):
            return 0

    policy_files.sort(key=get_step)

    if not policy_files:
        print(f"No policy-*.pkl files found in {log_dir}")
        sys.exit(0)

    # Prepare log file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = log_dir / f"log_{env_id}_{timestamp}.csv"
    
    print(f"Found {len(policy_files)} policies. Logging results to {log_file_path}")

    with open(log_file_path, mode='w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['step', 'avg_ret', 'std_ret'])

        for policy_path in policy_files:
            step = get_step(policy_path)
            
            with open(policy_path, "rb") as f_pkl:
                policy_params = pickle.load(f_pkl)
            
            ep_len_list, ep_ret_list = evaluate(env, policy_fn, policy_params, args.num_episodes)
            
            ep_ret = np.array(ep_ret_list)
            avg_ret = ep_ret.mean()
            std_ret = ep_ret.std()
            writer.writerow([step, avg_ret, std_ret])
            f_csv.flush()
            print(f"Step {step:8d}: avg_ret={avg_ret:10.2f}, std_ret={std_ret:10.2f}")

    print(f"\nEvaluation finished. Results saved to {log_file_path}")

if __name__ == "__main__":
    main()

