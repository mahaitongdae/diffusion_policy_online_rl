# Efficient Online Reinforcement Learning for Diffusion Policies

This is the official implementation of

**Efficient Online Reinforcement Learning for Diffusion Policies**

accepted by ICML 2025.

<p align="left">
<a href='https://arxiv.org/pdf/2502.00361' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/arXiv-PDF-red?style=flat&logo=arXiv&logoColor=wihte' alt='arXiv PDF'>
</a>
</p>

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

## Ackwonledgement
We developed this repo based on [DACER](https://github.com/happy-yan/DACER-Diffusion-with-Online-RL.git). We thank the authors of DACER for providing high-quality code base.

## Bibtex
If you used this repo in your paper, please considering 
giving us a star 🌟 and citing our related paper.

```bibtex
@article{ma2025soft,
  title={Efficient Online Reinforcement Learning for Diffusion Policy},
  author={Ma, Haitong and Chen, Tianyi and Wang, Kai and Li, Na and Dai, Bo},
  journal={arXiv preprint arXiv:2502.00361},
  year={2025}
}
```


