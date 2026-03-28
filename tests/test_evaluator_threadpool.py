"""
Test to demonstrate the thread pool effect in the evaluator.

Uses a mock evaluate function that sleeps to simulate work,
then compares sequential (1 worker) vs concurrent (4 workers) execution.

Run:
    python tests/test_evaluator_threadpool.py
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor


MOCK_EVAL_DURATION = 0.5  # seconds per eval


def mock_evaluate(step):
    """Simulates an evaluation that takes MOCK_EVAL_DURATION seconds."""
    tid = threading.get_ident()
    start = time.monotonic()
    time.sleep(MOCK_EVAL_DURATION)
    elapsed = time.monotonic() - start
    return {
        "step": step,
        "thread": tid,
        "avg_ret": step * 1.0,
        "std_ret": 0.1,
        "avg_len": 100.0,
        "elapsed": elapsed,
    }


def run_sequential(steps):
    """Run evaluations one at a time (simulates old behavior)."""
    results = []
    for step in steps:
        results.append(mock_evaluate(step))
    return results


def run_threadpool(steps, num_workers):
    """Run evaluations with a thread pool (simulates new behavior)."""
    results = []
    lock = threading.Lock()

    def worker(step):
        result = mock_evaluate(step)
        with lock:
            results.append(result)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(worker, step) for step in steps]
        for f in futures:
            f.result()
    return results


def test_threadpool_speedup():
    num_evals = 8
    num_workers = 4
    steps = list(range(1000, 1000 + num_evals * 1000, 1000))

    # Sequential
    t0 = time.monotonic()
    seq_results = run_sequential(steps)
    seq_time = time.monotonic() - t0

    # Concurrent
    t0 = time.monotonic()
    pool_results = run_threadpool(steps, num_workers)
    pool_time = time.monotonic() - t0

    print(f"{'='*60}")
    print(f"Mock eval duration: {MOCK_EVAL_DURATION}s  |  Num evals: {num_evals}")
    print(f"{'='*60}")

    print(f"\n--- Sequential (1 worker) ---")
    print(f"Total time: {seq_time:.2f}s  (expected ~{MOCK_EVAL_DURATION * num_evals:.1f}s)")
    for r in seq_results:
        print(f"  step={r['step']}  thread={r['thread']}  took={r['elapsed']:.2f}s")

    print(f"\n--- Thread Pool ({num_workers} workers) ---")
    print(f"Total time: {pool_time:.2f}s  (expected ~{MOCK_EVAL_DURATION * num_evals / num_workers:.1f}s)")
    unique_threads = set(r["thread"] for r in pool_results)
    print(f"Unique threads used: {len(unique_threads)}")
    for r in sorted(pool_results, key=lambda x: x["step"]):
        print(f"  step={r['step']}  thread={r['thread']}  took={r['elapsed']:.2f}s")

    speedup = seq_time / pool_time
    print(f"\n--- Result ---")
    print(f"Speedup: {speedup:.2f}x  (sequential {seq_time:.2f}s vs pool {pool_time:.2f}s)")

    assert pool_time < seq_time * 0.6, (
        f"Thread pool should be significantly faster: {pool_time:.2f}s vs {seq_time:.2f}s"
    )
    assert len(unique_threads) > 1, "Thread pool should use multiple threads"
    assert len(pool_results) == num_evals, "All evaluations should complete"
    print("All assertions passed!")


if __name__ == "__main__":
    test_threadpool_speedup()
