# Efficient Online Reinforcement Learning for Diffusion Policies

This is the official implementation of

**Efficient Online Reinforcement Learning for Diffusion Policies**

accepted by ICML 2025.

## Installation

```bash
# Create environment
conda create -n relax python=3.9 numpy tqdm tensorboardX matplotlib scikit-learn black snakeviz ipykernel setproctitle numba
conda activate relax

# One of: Install jax WITH CUDA 
pip install --upgrade "jax[cuda12]==0.4.27" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install package
pip install -r requirements.txt
pip install -e .
```



## Run
```bash
# Run one experiment
XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python scripts/train_mujoco.py --alg sdac --seed 100
```

## Visualize results
```python
from relax.utils.inspect_results import load_results, plot_mean

env_name = 'Ant-v4'

patterns_dict = {
        'sdac': r'sdac.*' # regex expression of saved folders
    }

for key, value in patterns_dict.items():
    print(key)
    _ = load_results(value, env_name, show_df=False)

plot_mean(patterns_dict, env_name)
```


