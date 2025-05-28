from collections import defaultdict
from typing import Callable

from numba import njit, types as nt
import numpy as np

class Accumulator:
    __slots__ = ("prefix", "buffer")

    def __init__(self, prefix=""):
        self.prefix = prefix
        self.buffer = defaultdict(list)

    def add(self, key, value):
        self.buffer[key].append(value)

    def add_vec(self, key, value):
        self.buffer[key].extend(value)

    def add_all(self, data: dict):
        for key, value in data.items():
            self.add(key, value)

    def reset(self):
        self.buffer.clear()

    def log(self, log_fn: Callable[[str, float], None]):
        for key, values in self.buffer.items():
            key = key if not self.prefix else f"{self.prefix}/{key}"
            value = sum(values) / len(values)
            log_fn(key, value)

class SampleLog:
    __slots__ = ("sample_step", "sample_episode", "episode_return", "episode_length", "accumulator")

    def __init__(self):
        self.sample_step = 0
        self.sample_episode = 0
        self.episode_return = 0.0
        self.episode_length = 0
        self.accumulator = Accumulator("sample")

    def add(self, reward: float, terminated: bool, truncated: bool, info: dict):
        self.episode_return += reward
        self.episode_length += 1
        self.sample_step += 1

        done = terminated or truncated
        if done:
            self.sample_episode += 1
            self.accumulator.add("episode_return", float(self.episode_return))
            self.accumulator.add("episode_length", self.episode_length)
            self.episode_return = 0.0
            self.episode_length = 0

        return done

    def log(self, log_fn: Callable[[str, float, int], None]):
        self.accumulator.log(lambda k, v: log_fn(k, v, self.sample_step))
        self.accumulator.reset()


class VectorSampleLog:
    __slots__ = ("num_envs", "sample_step", "sample_episode", "episode_return", "episode_length", "accumulator")

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.sample_step = 0
        self.sample_episode = 0
        self.episode_return = np.zeros((num_envs,), dtype=np.float64)
        self.episode_length = np.zeros((num_envs,), dtype=np.int64)
        self.accumulator = Accumulator("sample")

    def add(self, reward: np.ndarray, terminated: np.ndarray, truncated: np.ndarray, info: dict):
        self.episode_return += reward
        self.episode_length += 1
        self.sample_step += self.num_envs

        done = terminated | truncated
        done_count = np.count_nonzero(done)

        self.sample_episode += done_count
        self.accumulator.add_vec("episode_return", self.episode_return[done].tolist())
        self.accumulator.add_vec("episode_length", self.episode_length[done].tolist())
        self.episode_return[done] = 0.0
        self.episode_length[done] = 0

        return done_count > 0

    def log(self, log_fn: Callable[[str, float, int], None]):
        self.accumulator.log(lambda k, v: log_fn(k, v, self.sample_step))
        self.accumulator.reset()

class VectorFragmentSampleLog:
    __slots__ = ("num_envs", "fragment_length", "sample_step", "sample_episode", "episode_return", "episode_length", "accumulator")

    def __init__(self, num_envs: int, fragment_length: int):
        self.num_envs = num_envs
        self.fragment_length = fragment_length
        self.sample_step = 0
        self.sample_episode = 0
        self.episode_return = np.zeros((num_envs,), dtype=np.float64)
        self.episode_length = np.zeros((num_envs,), dtype=np.int64)
        self.accumulator = Accumulator("sample")

    def add(self, reward: np.ndarray, terminated: np.ndarray, truncated: np.ndarray, info: dict):
        done_count, complete_episode_return, complete_episode_length = process_fragment(reward, terminated, truncated, self.episode_return, self.episode_length, self.num_envs, self.fragment_length)
        self.sample_step += self.num_envs * self.fragment_length
        self.sample_episode += done_count
        self.accumulator.add_vec("episode_return", complete_episode_return.tolist())
        self.accumulator.add_vec("episode_length", complete_episode_length.tolist())
        return done_count > 0

    def log(self, log_fn: Callable[[str, float, int], None]):
        self.accumulator.log(lambda k, v: log_fn(k, v, self.sample_step))
        self.accumulator.reset()

@njit([(nt.float64[:, ::1], nt.boolean[:, ::1], nt.boolean[:, ::1], nt.float64[::1], nt.int64[::1], nt.int64, nt.int64)], cache=True)
def process_fragment(reward: np.ndarray, terminated: np.ndarray, truncated: np.ndarray, episode_return: np.ndarray, episode_length: np.ndarray, num_envs: int, fragment_length: int):
    assert reward.shape == terminated.shape == truncated.shape == (num_envs, fragment_length)

    done = terminated | truncated
    done_count = np.count_nonzero(done)

    if done_count > 0:
        complete_episode_return = np.empty((done_count,), dtype=np.float64)
        complete_episode_length = np.empty((done_count,), dtype=np.int64)
        ptr = 0
        for i in range(num_envs):
            initial_return = episode_return[i]
            initial_length = episode_length[i]
            left = 0
            for j in range(fragment_length):
                if done[i, j]:
                    right = j + 1
                    complete_episode_return[ptr] = reward[i, left:right].sum() + initial_return
                    complete_episode_length[ptr] = right - left + initial_length
                    ptr += 1
                    left = right
                    initial_return = 0.0
                    initial_length = 0
            episode_return[i] = initial_return + reward[i, left:].sum()
            episode_length[i] = fragment_length - left + initial_length
    else:
        episode_return += reward.sum(axis=-1)
        episode_length += fragment_length

    return done_count, complete_episode_return, complete_episode_length

class UpdateLog:
    __slots__ = ("update_step", "accumulator")

    def __init__(self):
        self.update_step = 0
        self.accumulator = Accumulator("update")

    def add(self, metrics: dict):
        self.update_step += 1
        self.accumulator.add_all(metrics)

    def log(self, log_fn: Callable[[str, float, int], None]):
        self.accumulator.log(lambda k, v: log_fn(k, v, self.update_step * 5))
        self.accumulator.reset()


class Interval:
    __slots__ = ("interval", "last_step")

    def __init__(self, interval: int):
        self.interval = interval
        self.last_step = 0

    def check(self, step: int) -> bool:
        if step - self.last_step >= self.interval:
            self.last_step = step
            return True
        return False
