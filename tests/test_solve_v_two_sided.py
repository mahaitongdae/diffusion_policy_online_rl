"""
Test solve_v_batch_two_sided against brute-force verification.

Run:
    python tests/test_solve_v_two_sided.py
"""

import jax
import jax.numpy as jnp
import numpy as np

import sys
sys.path.insert(0, ".")
from relax.algorithm.dpmdv2_fix12345_bon_noise_n4_ub import solve_v_batch_two_sided, solve_v_batch


def verify(x, l, v, lower_bound, upper_bound):
    """Compute mean(clip((x - v) / l, lower_bound, upper_bound)) and check it equals 1."""
    raw = (x - v) / l
    clipped = jnp.clip(raw, lower_bound, upper_bound)
    return jnp.mean(clipped, axis=-1, keepdims=True)


def test_reduces_to_one_sided():
    """With upper_bound=inf, should match solve_v_batch."""
    np.random.seed(42)
    B, N = 16, 32
    x = jnp.array(np.random.randn(B, N))
    l = jnp.array(np.random.uniform(0.5, 2.0, (B, 1)))
    lb = -0.5

    v_one_sided = solve_v_batch(x, l, lower_bound=lb)
    v_two_sided = solve_v_batch_two_sided(x, l, lower_bound=lb, upper_bound=jnp.inf)

    diff = jnp.abs(v_one_sided - v_two_sided).max()
    mean_val = verify(x, l, v_two_sided, lb, jnp.inf)
    err = jnp.abs(mean_val - 1.0).max()

    print(f"test_reduces_to_one_sided: v_diff={diff:.2e}, constraint_err={err:.2e}")
    assert diff < 1e-4, f"v mismatch: {diff}"
    assert err < 1e-4, f"constraint violated: {err}"
    print("  PASS")


def test_two_sided_basic():
    """Basic test with finite upper bound."""
    np.random.seed(123)
    B, N = 32, 16
    x = jnp.array(np.random.randn(B, N) * 3)
    l = jnp.array(np.random.uniform(0.5, 2.0, (B, 1)))
    lb, ub = -0.3, 2.0

    v = solve_v_batch_two_sided(x, l, lower_bound=lb, upper_bound=ub)
    mean_val = verify(x, l, v, lb, ub)
    err = jnp.abs(mean_val - 1.0).max()

    print(f"test_two_sided_basic: constraint_err={err:.2e}")
    assert err < 1e-4, f"constraint violated: {err}"
    print("  PASS")


def test_two_sided_tight():
    """Tight bounds that clip many items."""
    np.random.seed(456)
    B, N = 16, 32
    x = jnp.array(np.random.randn(B, N) * 5)
    l = jnp.array(np.ones((B, 1)))
    lb, ub = -0.5, 1.5

    v = solve_v_batch_two_sided(x, l, lower_bound=lb, upper_bound=ub)
    mean_val = verify(x, l, v, lb, ub)
    err = jnp.abs(mean_val - 1.0).max()

    print(f"test_two_sided_tight: constraint_err={err:.2e}")
    assert err < 1e-4, f"constraint violated: {err}"
    print("  PASS")


def test_weights_are_clipped():
    """Verify that resulting weights are within [lb, ub]."""
    np.random.seed(789)
    B, N = 8, 32
    x = jnp.array(np.random.randn(B, N) * 4)
    l = jnp.array(np.ones((B, 1)) * 1.5)
    lb, ub = -0.3, 2.5

    v = solve_v_batch_two_sided(x, l, lower_bound=lb, upper_bound=ub)
    weights = jnp.clip((x - v) / l, lb, ub)

    assert weights.min() >= lb - 1e-6, f"weights below lower bound: {weights.min()}"
    assert weights.max() <= ub + 1e-6, f"weights above upper bound: {weights.max()}"
    mean_err = jnp.abs(weights.mean(axis=-1) - 1.0).max()
    assert mean_err < 1e-4, f"mean constraint violated: {mean_err}"
    print(f"test_weights_are_clipped: weights in [{weights.min():.4f}, {weights.max():.4f}], mean_err={mean_err:.2e}")
    print("  PASS")


if __name__ == "__main__":
    test_reduces_to_one_sided()
    test_two_sided_basic()
    test_two_sided_tight()
    test_weights_are_clipped()
    print("\nAll tests passed!")
