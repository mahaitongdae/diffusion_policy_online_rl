"""
Benchmark solve_v_batch vs solve_v_batch_two_sided.

Run:
    python tests/bench_solve_v.py
"""

import time
import jax
import jax.numpy as jnp
import numpy as np

import sys
sys.path.insert(0, ".")
from relax.algorithm.dpmdv2_fix12345_bon_noise_n4_ub import solve_v_batch, solve_v_batch_two_sided


def bench(fn, x, l, kwargs, num_warmup=5, num_iters=200):
    jitted = jax.jit(lambda x, l: fn(x, l, **kwargs))

    # Warmup (compile + cache)
    for _ in range(num_warmup):
        v = jitted(x, l).block_until_ready()

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(num_iters):
        v = jitted(x, l).block_until_ready()
    elapsed = time.perf_counter() - t0

    return elapsed / num_iters, v


def main():
    np.random.seed(42)

    configs = [
        {"B": 256, "N": 16},
        {"B": 256, "N": 32},
        {"B": 256, "N": 64},
        {"B": 512, "N": 32},
    ]

    lb, ub = -0.5, 2.0

    print(f"{'Config':<16} {'one-sided (ms)':>15} {'two-sided (ms)':>15} {'two-sided ub=inf (ms)':>22} {'ratio':>8}")
    print("-" * 80)

    for cfg in configs:
        B, N = cfg["B"], cfg["N"]
        x = jnp.array(np.random.randn(B, N).astype(np.float32))
        l = jnp.array(np.random.uniform(0.5, 2.0, (B, 1)).astype(np.float32))

        t1, v1 = bench(solve_v_batch, x, l, {"lower_bound": lb})
        t2, v2 = bench(solve_v_batch_two_sided, x, l, {"lower_bound": lb, "upper_bound": ub})
        t3, v3 = bench(solve_v_batch_two_sided, x, l, {"lower_bound": lb, "upper_bound": jnp.inf})

        # Verify correctness
        diff_v = jnp.abs(v1 - v3).max()

        # Check normalization: mean(clip((x - v) / l, lb, ub)) should equal 1
        weights_1 = jnp.clip((x - v1) / l, lb, jnp.inf)
        weights_2 = jnp.clip((x - v2) / l, lb, ub)
        weights_3 = jnp.clip((x - v3) / l, lb, jnp.inf)
        mean_1 = weights_1.mean(axis=-1)
        mean_2 = weights_2.mean(axis=-1)
        mean_3 = weights_3.mean(axis=-1)
        err_1 = jnp.abs(mean_1 - 1.0).max()
        err_2 = jnp.abs(mean_2 - 1.0).max()
        err_3 = jnp.abs(mean_3 - 1.0).max()

        label = f"B={B}, N={N}"
        print(f"{label:<16} {t1*1000:>15.3f} {t2*1000:>15.3f} {t3*1000:>22.3f} {t2/t1:>7.1f}x")
        print(f"{'':16} norm_err: one-sided={err_1:.2e}  two-sided={err_2:.2e}  two-sided(inf)={err_3:.2e}  v_diff(1v3)={diff_v:.2e}")

    print()
    print("one-sided    = solve_v_batch (lower bound only)")
    print("two-sided    = solve_v_batch_two_sided (lower + upper bound)")
    print("two-sided ub=inf = solve_v_batch_two_sided with upper_bound=inf (should match one-sided result)")


if __name__ == "__main__":
    main()
