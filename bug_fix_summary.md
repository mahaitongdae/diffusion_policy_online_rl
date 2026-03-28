# DPMDv2 Bug Fix Summary (fix12345)

## Bug #1: Q-Network Updated Every Step Instead of Delayed

**File:** `dpmdv2_fix1.py` | **Location:** `stateless_update`, Q-param update lines

The original code unconditionally applied Q-network parameter updates at every step:
```python
q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)
```
**Fix:** Removed these lines so Q-networks are only updated through `delay_param_update`, respecting the `delay_update` schedule (typically every 2 steps).

---

## Bug #2: Alpha Loss/Grad Unpacking Order Swapped

**File:** `dpmdv2_fix12.py` | **Location:** `stateless_update`, alpha loss computation

`jax.value_and_grad()` returns `(value, grad)`, but the code unpacked as `(grad, value)`:
```python
# Bug:
alpha_grad, alpha_loss = jax.value_and_grad(alpha_loss_fn)(alpha_variable)
# Fix:
alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(alpha_variable)
```
**Fix:** Corrected unpacking order so loss and gradient are assigned to the right variables.

---

## Bug #3: Noise Scale Loss Linear Instead of Squared + Unpacking Swapped

**File:** `dpmdv2_fix123.py` | **Location:** `stateless_update`, noise scale loss

Two issues:
1. Loss was linear (`exp(x) - target`) instead of squared (`(exp(x) - target)^2`), providing weak optimization signal.
2. Same `value_and_grad` unpacking bug as #2.

```python
# Bug:
def noise_scale_loss_fn(log_noise_scale):
    return jnp.exp(log_noise_scale) - self.target_noise_scale
noise_scale_grad, noise_scale_loss = jax.value_and_grad(noise_scale_loss_fn)(log_noise_scale)

# Fix:
def noise_scale_loss_fn(log_noise_scale):
    return (jnp.exp(log_noise_scale) - self.target_noise_scale) ** 2
noise_scale_loss, noise_scale_grad = jax.value_and_grad(noise_scale_loss_fn)(log_noise_scale)
```

---

## Bug #4: Diffusion Noise Key Reused Across Particles + Incorrect vmap

**File:** `dpmdv2_fix1234.py` | **Location:** `stateless_update`, key splitting + policy loss vmap

Two issues:
1. A single `diffusion_noise_key` was shared across all particles in `vmap`, so every particle got identical noise during denoising — destroying gradient diversity.
2. `key` and `model` were bound via `partial()` instead of passed as positional args with explicit `in_axes`, causing incorrect vmap behavior.

**Fix:** Split the noise key into one per particle and restructured the vmap call:
```python
diffusion_noise_keys = jax.random.split(diffusion_noise_key, self.agent.num_particles)
# ...
loss, loss_info = jax.vmap(loss_fn, in_axes=(0, 0, None, 0, 0))(
    diffusion_noise_keys, q_weights, denoiser, t, batch_action)
```

---

## Bug #5 (this file): Out-of-Range Actions After Noise Addition

**File:** `dpmdv2_fix12345.py` | **Location:** Monkey-patched `get_action` and `get_batch_action_with_q`

In `diffv4.py`, exploration noise is added **after** clipping actions to `[-1, 1]`, so the resulting actions can exceed the valid range. These out-of-range actions are then fed to Q-networks for best-of-N selection and weight computation, producing Q estimates on inputs the networks were never trained on.

**Fix:** Monkey-patch both `get_action` and `get_batch_action_with_q` to clip actions to `[-1, 1]` after noise addition but before Q evaluation:
```python
# Original (in diffv4.py):
acts = jax.vmap(sample)(keys)                                          # clipped to [-1, 1]
acts = acts + jax.random.normal(noise_key, acts.shape) * jnp.exp(log_noise_scale)  # can exceed [-1, 1]
qs = jax.vmap(q_fn)(acts)                                              # Q on out-of-range actions

# Fix (monkey-patched):
acts = jax.vmap(sample)(keys)                                          # clipped to [-1, 1]
acts = (acts + jax.random.normal(noise_key, acts.shape) * jnp.exp(log_noise_scale)).clip(-1, 1)
qs = jax.vmap(q_fn)(acts)                                              # Q on valid actions
```

---

## Summary Table

| Fix | Bug | Root Cause | Severity |
|-----|-----|-----------|----------|
| #1 | Q-networks updated every step | Missing delay condition | High — breaks delayed update schedule |
| #2 | Alpha loss/grad swapped | `value_and_grad` unpacking error | High — wrong gradient used for updates |
| #3 | Noise scale loss too weak + swapped | Linear loss + unpacking error | Medium — poor noise scale convergence |
| #4 | Identical noise across particles | Key reuse + bad vmap config | High — correlated gradients, poor training |
| #5 | Actions exceed [-1, 1] after noise | No clip after noise addition | Low-Medium — Q extrapolation on OOD inputs |
