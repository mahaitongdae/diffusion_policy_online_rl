from contextlib import ContextDecorator
import time


class catchtime(ContextDecorator):
    def __init__(self, event: str):
        super().__init__()
        self.event = event

    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, *exc):
        self.end = time.perf_counter_ns()
        self.interval = self.end - self.start
        print(f"{self.event} took {self.interval / 1e6:.3f} ms")
        return False
