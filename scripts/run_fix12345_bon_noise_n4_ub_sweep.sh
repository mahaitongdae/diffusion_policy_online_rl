#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
GPUS=(0 1 2 3)
SEEDS=(100 101 102 103 104)
NUM_EACH_GPU=1

PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

GROUP_NAME=bon_noise_n4_ub_sweep
ENV_NAMES=(Ant-v4 Walker2d-v4)

run_task() {
    local env_name=$1
    local seed=$2
    local slot=$3
    local num_gpus=${#GPUS[@]}
    local device_idx=$((slot % num_gpus))
    local device=${GPUS[$device_idx]}

    export CUDA_VISIBLE_DEVICES=$device
    export XLA_PYTHON_CLIENT_PREALLOCATE=false

    mkdir -p "./scripts/logs/${GROUP_NAME}"
    local timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

    echo "GPU $device: Running env=$env_name seed=$seed"

    python ./scripts/train_mujoco_1m.py \
        --alg dpmdv2_fix12345_bon_noise_n4_ub \
        --env "$env_name" \
        --reweight_type negative_strictly_normalized_relu_linear \
        --wandb_group "${GROUP_NAME}" \
        --kl_constraint 1.5 \
        --learnable_alpha \
        --noise_scale_lr 7e-3 \
        --clip_lower_bound -0.5 \
        --clip_upper_bound 2.0 \
        --regularization_type clipped_only \
        --negative_weights_regularization 1 \
        --bellman_next_action_policy target \
        --batch_action_policy target \
        --save_to_shared_folder \
        --seed "$seed" \
        --suffix "bon_noise_n4_ub2.0_lb-0.5" \
        > "./scripts/logs/${GROUP_NAME}/gpu_${device}_${env_name}_s${seed}_${timestamp}.txt" 2>&1

    echo "GPU $device: Completed env=$env_name seed=$seed"
}

echo "Running bon_noise_n4_ub: envs ${ENV_NAMES[*]}, seeds ${SEEDS[*]} on GPUs ${GPUS[*]}"
echo "Starting at $(date)"

. env_parallel.bash
env_parallel --bar -P${PARALLEL} run_task {1} {2} {%} ::: ${ENV_NAMES[@]} ::: ${SEEDS[@]}

echo "All done at $(date)"
