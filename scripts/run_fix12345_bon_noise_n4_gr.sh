#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
GPUS=(0 1)
SEEDS=(100)
NUM_EACH_GPU=1

PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

GROUP_NAME=group_relative_linear_sweep
ENV_NAMES=(Ant-v4 Walker2d-v4)
CLIP_BOUNDS=(0.0 -0.1)

run_task() {
    local env_name=$1
    local clip_lb=$2
    local seed=$3
    local slot=$4
    local num_gpus=${#GPUS[@]}
    local device_idx=$((slot % num_gpus))
    local device=${GPUS[$device_idx]}

    export CUDA_VISIBLE_DEVICES=$device
    export XLA_PYTHON_CLIENT_PREALLOCATE=false

    mkdir -p "./scripts/logs/${GROUP_NAME}"
    local timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

    echo "GPU $device: Running env=$env_name clip_lb=$clip_lb seed=$seed"

    python ./scripts/train_mujoco_1m.py \
        --alg dpmdv2_fix12345_bon_noise_n4 \
        --env "$env_name" \
        --reweight_type group_relative_linear \
        --wandb_group "${GROUP_NAME}" \
        --noise_scale_lr 7e-3 \
        --clip_lower_bound "$clip_lb" \
        --bellman_next_action_policy target \
        --batch_action_policy target \
        --save_to_shared_folder \
        --seed "$seed" \
        --suffix "bon_noise_n4_gr_running_stats_clip_lb${clip_lb}" \
        > "./scripts/logs/${GROUP_NAME}/gpu_${device}_${env_name}_clip${clip_lb}_s${seed}_${timestamp}.txt" 2>&1

    echo "GPU $device: Completed env=$env_name clip_lb=$clip_lb seed=$seed"
}

echo "Running bon_noise_n4 clip_lower_bound sweep: envs ${ENV_NAMES[*]}, clips ${CLIP_BOUNDS[*]}, seeds ${SEEDS[*]} on GPUs ${GPUS[*]}"
echo "Starting at $(date)"

. env_parallel.bash
env_parallel --bar -P${PARALLEL} run_task {1} {2} {3} {%} ::: ${ENV_NAMES[@]} ::: ${CLIP_BOUNDS[@]} ::: ${SEEDS[@]}

echo "All done at $(date)"
