#!/bin/bash

# Installation script for Diffusion Policy Online RL
# Usage: bash install.sh

set -e  # Exit on error

echo "========================================="
echo "Installing Diffusion Policy Online RL"
echo "========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Create conda environment
echo ""
echo "Step 1/4: Creating conda environment 'relax'..."
conda create -n relax python=3.9 numpy tqdm tensorboardX matplotlib scikit-learn black snakeviz ipykernel setproctitle numba -y

echo ""
echo "Step 2/4: Installing JAX with CUDA 12 support..."
conda run -n relax pip install --upgrade "jax[cuda12]==0.4.27" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo ""
echo "Step 3/4: Installing dependencies from requirements.txt..."
conda run -n relax pip install -r requirements.txt

echo ""
echo "Step 4/4: Installing package in editable mode..."
conda run -n relax pip install -e .

echo ""
echo "========================================="
echo "Installation completed successfully!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate relax"
echo ""
echo "To test the installation, run:"
echo "  XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python scripts/train_mujoco.py --alg sdac --seed 100"
echo ""
