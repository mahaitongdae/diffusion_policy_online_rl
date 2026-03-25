import argparse
import os.path
import sys
from pathlib import Path
import time
from functools import partial
import yaml
import math
import jax, jax.numpy as jnp

from relax.algorithm.dpmd_v2 import DPMDV2
from relax.buffer import TreeBuffer
from relax.network.diffv4 import create_diffv4_net
from relax.trainer.off_policy import OffPolicyTrainer
from relax.env import create_env, create_vector_env
from relax.utils.experience import Experience
from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding
from relax.utils.log_diff import log_git_details

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="dpmdv2")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--suffix", type=str, default="debug")
    parser.add_argument("--num_vec_envs", type=int, default=0)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--wandb_project_name", type=str, default="diffusion_online_rl_negative")
    parser.add_argument("--start_step", type=int, default=int(3e4)) # other envs 3e4
    parser.add_argument("--total_step", type=int, default=int(1e6))
    parser.add_argument("--cluster", default=False, action="store_true")
    parser.add_argument("--save_to_shared_folder", default=False, action="store_true")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--eval_env", type=str, default='None')
    parser.add_argument("--wandb_group", type=str, default="debug")
    # network
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--num_particles", type=int, default=32)
    parser.add_argument("--num_best_of_n", type=int, default=32)
    parser.add_argument("--beta_schedule_scale", type=float, default=1.0)
    parser.add_argument("--beta_schedule_type", type=str, default='cosine')
    parser.add_argument("--initial_alpha", type=float, default=1e-4)
    parser.add_argument("--initial_noise_scale", type=float, default=0.5)
    # learning rate
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_schedule_end", type=float, default=3e-5)
    parser.add_argument("--lr_schedule_steps", type=int, default=int(5e5))
    parser.add_argument("--lr_schedule_begin", type=int, default=int(2.5e5))
    # alpha
    parser.add_argument("--learnable_alpha", default=False, action='store_true')
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--kl_constraint", type=float, default=1.5)
    # noise scale
    parser.add_argument("--learnable_noise_scale", default=False, action='store_true')
    parser.add_argument("--delay_log_noise_scale_update", type=int, default=1250)
    parser.add_argument("--noise_scale_lr", type=float, default=7e-3)
    parser.add_argument("--target_noise_scale", type=float, default=0.1)
    # reweighting
    parser.add_argument("--reweight_type", type=str, default='logsumexp')  # 'exp', 'none'
    parser.add_argument("--clipped_lower_bound", type=float, default=0.0)
    parser.add_argument("--negative_weights_regularization", type=float, default=0.0)
    parser.add_argument("--regularization_type", type=str, default='square')
    parser.add_argument("--clipped_only_weighted_mse_lower_bound", type=float, default=-1.0)
    parser.add_argument("--add_state_level_reweighting", default=False, action='store_true')
    parser.add_argument("--use_timestep_weight", default=False, action='store_true')
    parser.add_argument(
        "--bellman_next_action_policy",
        type=str,
        default="target",
        choices=("online", "target"),
        help="Which policy params to use for next-action sampling in the Bellman target (PEV): online or slow target.",
    )
    parser.add_argument(
        "--batch_action_policy",
        type=str,
        default="target",
        choices=("online", "target"),
        help="Which policy params to use for get_batch_action_with_q reweighting samples: online or slow target.",
    )
    args = parser.parse_args()

    if args.debug:
        from jax import config
        config.update("jax_disable_jit", True)
        
    if 'dm_control' in args.env:
        from relax.env.dmc.register import register_dm_control_envs
        register_dm_control_envs()

    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    env_seed, env_action_seed, eval_env_seed, buffer_seed, init_network_seed, train_seed = map(
        int, master_rng.integers(0, 2**32 - 1, 6)
    )
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)
    del init_network_seed, train_seed

    if args.num_vec_envs > 0:
        env, obs_dim, act_dim = create_vector_env(args.env, args.num_vec_envs, env_seed, env_action_seed, mode="futex")
    else:
        env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed)
    if args.eval_env != 'None':
        eval_env, _, _ = create_env(args.eval_env, eval_env_seed, env_action_seed)
    else:
        eval_env = env


    buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=int(1e6), seed=buffer_seed)

    print(f"Algorithm: {args.alg}")

    def mish(x: jax.Array):
        return x * jnp.tanh(jax.nn.softplus(x))
    
    agent, params = create_diffv4_net(
        init_network_key, 
        obs_dim, act_dim, 
        [args.hidden_dim] * args.hidden_num, 
        [args.diffusion_hidden_dim] * args.hidden_num, 
        mish,
        num_timesteps=args.diffusion_steps, 
        num_particles=args.num_particles, 
        num_best_of_n=args.num_best_of_n,
        beta_schedule_scale=args.beta_schedule_scale,
        beta_schedule_type=args.beta_schedule_type,
        initial_alpha=args.initial_alpha,
        initial_noise_scale=args.initial_noise_scale
    )
    algorithm = DPMDV2(
        agent, params, 
        lr=args.lr, 
        lr_schedule_end=args.lr_schedule_end,
        lr_schedule_steps=args.lr_schedule_steps,
        lr_schedule_begin=args.lr_schedule_begin,
        learnable_alpha=args.learnable_alpha,
        alpha_lr=args.alpha_lr, 
        kl_constraint=args.kl_constraint,
        learnable_noise_scale=args.learnable_noise_scale,
        delay_log_noise_scale_update=args.delay_log_noise_scale_update,
        noise_scale_lr=args.noise_scale_lr,
        target_noise_scale=args.target_noise_scale,
        reweight_type=args.reweight_type,
        clipped_lower_bound=args.clipped_lower_bound,
        negative_weights_regularization=args.negative_weights_regularization,
        regularization_type=args.regularization_type,
        clipped_only_weighted_mse_lower_bound=args.clipped_only_weighted_mse_lower_bound,
        add_state_level_reweighting=args.add_state_level_reweighting,
        use_timestep_weight=args.use_timestep_weight,
        bellman_next_action_use_target_policy=(
            args.bellman_next_action_policy == "target"
        ),
        policy_batch_action_use_target_policy=(
            args.batch_action_policy == "target"
        )
    )

    if args.cluster:
        PROJECT_ROOT = Path('/n/netscratch/nali_lab_seas/Lab/haitongma/sdac_logs')
    if args.save_to_shared_folder:
        os.makedirs('/mnt/shared/haitongma/data/diffusion_online_rl', exist_ok=True)
        PROJECT_ROOT = Path('/mnt/shared/haitongma/data/diffusion_online_rl')
    if args.logdir:
        PROJECT_ROOT = Path(args.logdir)
    
    exp_dir = PROJECT_ROOT / "logs" / args.wandb_group / args.env / (args.alg + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + f'_s{args.seed}_{args.suffix}')
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the command to a text file
    with open(os.path.join(exp_dir, 'command.txt'), 'w') as f:
        f.write(' '.join(sys.argv))

    # Save the arguments to a YAML file
    args_dict = vars(args)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file)

    log_git_details(log_file=os.path.join(exp_dir, 'dacer.diff'))

    trainer = OffPolicyTrainer(
        env=env,
        algorithm=algorithm,
        buffer=buffer,
        start_step=args.start_step,
        total_step=args.total_step,
        sample_per_iteration=1,
        evaluate_env=eval_env,
        save_policy_every=int(args.total_step / 20),
        warmup_with="random",
        log_path=exp_dir,
        update_log_n_step=1 if args.debug else 1000,
        hparams=args_dict,
        wandb_group=args.wandb_group,
        wandb_project_name=args.wandb_project_name
    )

    trainer.setup(Experience.create_example(obs_dim, act_dim, trainer.batch_size))
    
    trainer.run(train_key)
