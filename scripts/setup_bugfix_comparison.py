#!/usr/bin/env python3
"""
Setup script to create 4 versions of dpmd_v2.py with incremental bug fixes.
"""

import os
import shutil
from pathlib import Path

def apply_fix1(lines):
    """Fix 1: Remove duplicate Q-network updates (lines 545-546)"""
    # Remove lines 545-546: duplicate q1_params and q2_params updates
    result = []
    for i, line in enumerate(lines, 1):
        # Skip lines 545-546
        if i in [545, 546]:
            continue
        result.append(line)
    return result

def apply_fix2(lines):
    """Fix 2: Fix alpha value/grad swap (line 519)"""
    result = []
    for i, line in enumerate(lines, 1):
        if i == 519 and 'alpha_grad, alpha_loss = jax.value_and_grad' in line:
            # Swap the assignment
            result.append(line.replace(
                'alpha_grad, alpha_loss = jax.value_and_grad',
                'alpha_loss, alpha_grad = jax.value_and_grad'
            ))
        else:
            result.append(line)
    return result

def apply_fix3(lines):
    """Fix 3: Fix noise scale value/grad swap and loss function (lines 562-565 in original, 560-563 after fix1)"""
    result = []
    for i, line in enumerate(lines, 1):
        # Fix the loss function (line 561 after fix1 removed 2 lines)
        if i == 561 and 'return jnp.exp(log_noise_scale) - self.target_noise_scale' in line:
            result.append(line.replace(
                'return jnp.exp(log_noise_scale) - self.target_noise_scale',
                'return (jnp.exp(log_noise_scale) - self.target_noise_scale) ** 2'
            ))
        # Fix the swap (line 563 after fix1 removed 2 lines)
        elif i == 563 and 'noise_scale_grad, noise_scale_loss = jax.value_and_grad' in line:
            result.append(line.replace(
                'noise_scale_grad, noise_scale_loss = jax.value_and_grad',
                'noise_scale_loss, noise_scale_grad = jax.value_and_grad'
            ))
        else:
            result.append(line)
    return result

def apply_fix4(lines):
    """Fix 4: Split diffusion noise key for each particle to get diverse gradients"""
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Fix 1: Add key splitting after diffusion_noise_key is created (line ~241)
        if 'next_eval_key, new_eval_key, diffusion_time_key, diffusion_noise_key = jax.random.split(' in line:
            result.append(line)
            # Add the next line (key, 4))
            i += 1
            if i < len(lines):
                result.append(lines[i])
            # Add key splitting
            result.append('            # Bug #4 fix: Split noise key for each particle to get diverse gradients\n')
            result.append('            diffusion_noise_keys = jax.random.split(diffusion_noise_key, self.agent.num_particles)\n')
            i += 1
            continue

        # Fix 2: Update loss_fn partial to not include key or model (around line 497-509)
        if 'loss_fn = partial(' in line and 'self.agent.diffusion.weighted_p_loss' in lines[i+1]:
            # Replace the entire block
            result.append('                # Bug #4 fix: Bind only keyword-only args in partial, pass all positional args via vmap\n')
            result.append('                loss_fn = partial(\n')
            result.append('                    self.agent.diffusion.weighted_p_loss,\n')
            result.append('                    negative_weights_regularization=self.negative_weights_regularization,\n')
            result.append('                    regularization_type=self.regularization_type,\n')
            result.append('                    clipped_only_weighted_mse_lower_bound=self.clipped_only_weighted_mse_lower_bound,\n')
            result.append('                    use_timestep_weight=self.use_timestep_weight)\n')
            result.append('                # vmap over: key (0), weights (0), model (None), t (0), x_start (0)\n')
            result.append('                # Each particle gets: unique key, unique weights vector, shared model, unique timesteps, unique actions\n')
            result.append('                loss, loss_info = jax.vmap(loss_fn, in_axes=(0, 0, None, 0, 0))(\n')
            result.append('                    diffusion_noise_keys,  # Shape (32,) -> each particle gets unique key\n')
            result.append('                    jax.lax.stop_gradient(q_weights),  # Shape (32, 256) -> each particle gets weights[i]\n')
            result.append('                    denoiser,  # Broadcast - same model for all particles\n')
            result.append('                    t,  # Shape (32, 256) -> each particle gets t[i]\n')
            result.append('                    jax.lax.stop_gradient(batch_action))  # Shape (32, 256, act_dim) -> each particle gets batch_action[i]\n')

            # Skip until we find the closing of jax.vmap call
            while i < len(lines) and 'x_start=jax.lax.stop_gradient(batch_action))' not in lines[i]:
                i += 1
            i += 1
            continue

        result.append(line)
        i += 1

    return result

def rename_class(lines, old_name, new_name):
    """Rename the class in the file"""
    return [line.replace(f'class {old_name}', f'class {new_name}') for line in lines]

def patch_train_script():
    """Add imports and support for the new algorithm versions to train_mujoco_1m.py"""
    train_script = Path('scripts/train_mujoco_1m.py')

    with open(train_script, 'r') as f:
        lines = f.readlines()

    # Find and update the import line
    for i, line in enumerate(lines):
        if line.strip() == 'from relax.algorithm.dpmd_v2 import DPMDV2':
            lines[i] = line  # Keep original
            # Insert new imports right after
            lines.insert(i+1, 'from relax.algorithm.dpmdv2_current import DPMDV2Current\n')
            lines.insert(i+2, 'from relax.algorithm.dpmdv2_fix1 import DPMDV2Fix1\n')
            lines.insert(i+3, 'from relax.algorithm.dpmdv2_fix12 import DPMDV2Fix12\n')
            lines.insert(i+4, 'from relax.algorithm.dpmdv2_fix123 import DPMDV2Fix123\n')
            lines.insert(i+5, 'from relax.algorithm.dpmdv2_fix1234 import DPMDV2Fix1234\n')
            lines.insert(i+6, 'from relax.algorithm.dpmdv2_fix12345 import DPMDV2Fix12345\n')
            break

    # Find where to insert new algorithm blocks (right before elif args.alg == 'idem':)
    insertion_point = None
    for i, line in enumerate(lines):
        if line.strip() == "elif args.alg == 'idem':":
            insertion_point = i
            break

    if insertion_point:
        # Build the template for each algorithm variant
        template = """    elif args.alg == '{alg_name}':
        import math
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_diffv4_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=args.diffusion_steps,
                                          num_particles=args.num_particles,
                                          num_best_of_n=args.num_best_of_n,
                                          noise_scale=args.noise_scale,
                                          beta_schedule_scale=args.beta_schedule_scale,
                                          initial_alpha=args.init_alpha,
                                          alpha_transformation=args.alpha_transformation,
                                          initial_log_noise_scale=math.log(args.initial_noise_scale))
        algorithm = {class_name}(agent, params, lr=args.lr,
                           alpha_lr=args.alpha_lr,
                           lr_schedule_end=args.lr_schedule_end,
                           lr_schedule_steps=args.lr_schedule_steps,
                           lr_schedule_begin=args.lr_schedule_begin,
                           learnable_alpha=args.learnable_alpha,
                           kl_constraint=args.kl_constraint,
                           update_additive_noise_scale=args.update_additive_noise_scale,
                           reweight_type=args.reweight_type,
                           delay_log_noise_scale_update=args.delay_log_noise_scale_update,
                           clipped_lower_bound=args.clip_lower_bound,
                           negative_weights_regularization=args.negative_weights_regularization,
                           regularization_type=args.regularization_type,
                           clipped_only_weighted_mse_lower_bound=args.clipped_only_weighted_mse_lower_bound,
                           use_timestep_weight=args.use_timestep_weight,
                           target_noise_scale=args.target_noise_scale,
                           noise_scale_lr=args.noise_scale_lr,
                           add_state_level_reweighting=args.add_state_level_reweighting,
                           bellman_next_action_use_target_policy=(
                               args.bellman_next_action_policy == "target"
                           ),
                           policy_batch_action_use_target_policy=(
                               args.batch_action_policy == "target"
                           ))
"""

        variants = [
            ('dpmdv2_current', 'DPMDV2Current'),
            ('dpmdv2_fix1', 'DPMDV2Fix1'),
            ('dpmdv2_fix12', 'DPMDV2Fix12'),
            ('dpmdv2_fix123', 'DPMDV2Fix123'),
            ('dpmdv2_fix1234', 'DPMDV2Fix1234'),
            ('dpmdv2_fix12345', 'DPMDV2Fix12345'),
        ]

        # Insert in reverse order to maintain line numbers
        for alg_name, class_name in reversed(variants):
            block = template.format(alg_name=alg_name, class_name=class_name)
            lines.insert(insertion_point, block)

    # Write back
    with open(train_script, 'w') as f:
        f.writelines(lines)

    return train_script

def main():
    source_file = Path('relax/algorithm/dpmd_v2.py')
    base_dir = Path('relax/algorithm')

    if not source_file.exists():
        print(f"Error: {source_file} not found!")
        return

    print("="*50)
    print("Setting up bug fix comparison")
    print("="*50)

    # Read original file
    with open(source_file, 'r') as f:
        original_lines = f.readlines()

    # Create backup
    backup_file = source_file.with_suffix('.py.backup')
    print(f"\nCreating backup: {backup_file}")
    shutil.copy(source_file, backup_file)

    # Version 1: Current (no fixes)
    print("\nCreating dpmdv2_current.py (no fixes)...")
    current_lines = rename_class(original_lines, 'DPMDV2', 'DPMDV2Current')
    with open(base_dir / 'dpmdv2_current.py', 'w') as f:
        f.writelines(current_lines)

    # Version 2: Fix 1 only (double Q update)
    print("Creating dpmdv2_fix1.py (fix double Q update)...")
    fix1_lines = apply_fix1(original_lines)
    fix1_lines = rename_class(fix1_lines, 'DPMDV2', 'DPMDV2Fix1')
    with open(base_dir / 'dpmdv2_fix1.py', 'w') as f:
        f.writelines(fix1_lines)

    # Version 3: Fix 1+2 (double Q + alpha swap)
    print("Creating dpmdv2_fix12.py (fix double Q + alpha swap)...")
    fix12_lines = apply_fix1(original_lines)
    # After removing 2 lines, line 519 becomes 517
    fix12_lines = apply_fix2(fix12_lines)
    fix12_lines = rename_class(fix12_lines, 'DPMDV2', 'DPMDV2Fix12')
    with open(base_dir / 'dpmdv2_fix12.py', 'w') as f:
        f.writelines(fix12_lines)

    # Version 4: Fixes 1+2+3 (double Q + alpha swap + noise swap + loss)
    print("Creating dpmdv2_fix123.py (fixes 1+2+3)...")
    fix123_lines = apply_fix1(original_lines)
    fix123_lines = apply_fix2(fix123_lines)
    # After removing 2 lines, line 563/565 need adjustment
    fix123_lines = apply_fix3(fix123_lines)
    fix123_lines = rename_class(fix123_lines, 'DPMDV2', 'DPMDV2Fix123')
    with open(base_dir / 'dpmdv2_fix123.py', 'w') as f:
        f.writelines(fix123_lines)

    # Version 5: All fixes 1-4 (double Q + alpha swap + noise swap + loss + key split)
    print("Creating dpmdv2_fix1234.py (fixes 1-4)...")
    fix1234_lines = apply_fix1(original_lines)
    fix1234_lines = apply_fix2(fix1234_lines)
    fix1234_lines = apply_fix3(fix1234_lines)
    fix1234_lines = apply_fix4(fix1234_lines)
    fix1234_lines = rename_class(fix1234_lines, 'DPMDV2', 'DPMDV2Fix1234')
    with open(base_dir / 'dpmdv2_fix1234.py', 'w') as f:
        f.writelines(fix1234_lines)

    # Version 6: All fixes 1-5 (fixes 1-4 + diffusion initial noise scaling)
    print("Creating dpmdv2_fix12345.py (all fixes including diffusion noise)...")
    # Fix 5 is a monkey-patch applied in __init__, so we just copy fix1234 and rename
    # The monkey-patch code is already in the manually created file
    if not (base_dir / 'dpmdv2_fix12345.py').exists():
        print("  Warning: dpmdv2_fix12345.py not found, please create it manually from dpmdv2_fix1234.py")
        print("  Add the monkey-patch for Bug #5 in __init__ before @jax.jit")
    else:
        print("  Using existing dpmdv2_fix12345.py (manually created)")

    print("\nPatching train_mujoco_1m.py to support new algorithm versions...")
    patch_train_script()

    print("\n" + "="*50)
    print("Setup complete!")
    print("="*50)
    print("\nCreated files:")
    print(f"  - {base_dir}/dpmdv2_current.py (no fixes)")
    print(f"  - {base_dir}/dpmdv2_fix1.py (fix 1: double Q update)")
    print(f"  - {base_dir}/dpmdv2_fix12.py (fix 1+2: + alpha swap)")
    print(f"  - {base_dir}/dpmdv2_fix123.py (fix 1+2+3: + noise swap + loss)")
    print(f"  - {base_dir}/dpmdv2_fix1234.py (fix 1+2+3+4: + key split)")
    print(f"  - {base_dir}/dpmdv2_fix12345.py (all fixes + diffusion noise)")
    print(f"\nOriginal file backed up to: {backup_file}")
    print(f"Train script patched to support: dpmdv2_current, dpmdv2_fix1, dpmdv2_fix12, dpmdv2_fix123, dpmdv2_fix1234, dpmdv2_fix12345")
    print("\nNext: Run ./scripts/compare_bugfixes_parallel.sh")
    print("="*50)

if __name__ == '__main__':
    main()
