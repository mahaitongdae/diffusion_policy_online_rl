#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
GPUS=(0 1 2 3)
SEEDS=(100 101 102 103 104)
NUM_EACH_GPU=1

PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

GROUP_NAME=bellman_next_action_sweep_bon_32vs4
ENV_NAMES=(Walker2d-v4 Ant-v4)
ALGS=(dpmdv2_fix12345_bon_noise_n2 dpmdv2_fix12345_no_bon)

run_task() {
    local alg=$1
    local env_name=$2
    local seed=$3
    local slot=$4
    local num_gpus=${#GPUS[@]}
    local device_idx=$((slot % num_gpus))
    local device=${GPUS[$device_idx]}

    export CUDA_VISIBLE_DEVICES=$device
    export XLA_PYTHON_CLIENT_PREALLOCATE=false

    mkdir -p "./scripts/logs/${GROUP_NAME}"
    local timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

    echo "GPU $device: Running $alg env=$env_name seed=$seed"

    python ./scripts/train_mujoco_1m.py \
        --alg "$alg" \
        --env "$env_name" \
        --reweight_type negative_strictly_normalized_relu_linear \
        --wandb_group "${GROUP_NAME}" \
        --kl_constraint 1.5 \
        --learnable_alpha \
        --noise_scale_lr 7e-3 \
        --clip_lower_bound 0.0 \
        --regularization_type clipped_only \
        --negative_weights_regularization 0 \
        --bellman_next_action_policy target \
        --batch_action_policy target \
        --seed "$seed" \
        --suffix "${alg}_negative_strictly_normalized_relu_linear_kl1.5" \
        > "./scripts/logs/${GROUP_NAME}/gpu_${device}_${alg}_${env_name}_s${seed}_${timestamp}.txt" 2>&1

    echo "GPU $device: Completed $alg env=$env_name seed=$seed"
}

echo "Running bellman sweep: algs ${ALGS[*]}, envs ${ENV_NAMES[*]}, seeds ${SEEDS[*]} on GPUs ${GPUS[*]}"
echo "Starting at $(date)"

. env_parallel.bash
env_parallel --bar -P${PARALLEL} run_task {1} {2} {3} {%} ::: ${ALGS[@]} ::: ${ENV_NAMES[@]} ::: ${SEEDS[@]}

echo "All done at $(date)"
