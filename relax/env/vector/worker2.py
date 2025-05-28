import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json

import gymnasium
import numpy as np

import setproctitle
from relax.prctl import set_client_pdeathsig
from relax.spinlock import spinlock_client_wait, spinlock_client_notify

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

    i = args.index

    setproctitle.setproctitle(f"SpinlockWorker:{i}")
    set_client_pdeathsig()

    descr = json.loads(args.descr)
    action = initialize_shm(descr["action"], "r")
    obs = initialize_shm(descr["obs"], "r+")
    reward = initialize_shm(descr["reward"], "r+")
    terminated = initialize_shm(descr["terminated"], "r+")
    truncated = initialize_shm(descr["truncated"], "r+")
    signal = initialize_shm(descr["signal"], "r+")
    signal_pointer = signal.ctypes.data + i

    try:
        while True:
            spinlock_client_wait(signal_pointer)
            command = signal[i].item()
            if command == 1:
                obs[i], _ = env.reset()
                spinlock_client_notify(signal_pointer, -1)
            elif command == 2:
                obs[i], reward[i], terminated[i], truncated[i], _ = env.step(action[i])
                spinlock_client_notify(signal_pointer, -1)
            elif command == 3:
                env.close()
                break
            else:
                raise ValueError(f"Unknown command: {command}")
    except KeyboardInterrupt:
        pass
    finally:
        spinlock_client_notify(signal_pointer, -2)

if __name__ == "__main__":
    main()
