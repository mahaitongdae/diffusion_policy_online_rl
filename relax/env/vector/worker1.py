import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import os
import sys

import gymnasium
import numpy as np

import setproctitle
from relax.prctl import set_client_pdeathsig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--descr", type=str, required=True)
    return parser.parse_args()

def initialize_shm(descr, mode):
    arr = np.memmap(f"/dev/shm/{descr['name']}", dtype=descr["dtype"], mode=mode, shape=tuple(descr["shape"]))
    return arr

def main():
    args = parse_args()

    env = gymnasium.make(args.env)
    env.reset(seed=args.seed)

    rx = sys.stdin.fileno()
    tx = sys.stdout.fileno()
    i = args.index

    setproctitle.setproctitle(f"PipeWorker:{i}")
    set_client_pdeathsig()

    descr = json.loads(args.descr)
    action = initialize_shm(descr["action"], "r")
    obs = initialize_shm(descr["obs"], "r+")
    reward = initialize_shm(descr["reward"], "r+")
    terminated = initialize_shm(descr["terminated"], "r+")
    truncated = initialize_shm(descr["truncated"], "r+")

    try:
        while True:
            command = os.read(rx, 1)
            if command == b"r":
                obs[i], _ = env.reset()
                os.write(tx, b"o")
            elif command == b"s":
                obs[i], reward[i], terminated[i], truncated[i], _ = env.step(action[i])
                os.write(tx, b"o")
            elif command == b"q":
                env.close()
                break
            else:
                raise ValueError(f"Unknown command: {command}")
    except KeyboardInterrupt:
        pass
    finally:
        os.write(tx, b"q")
        os.close(rx)
        os.close(tx)

if __name__ == "__main__":
    main()
