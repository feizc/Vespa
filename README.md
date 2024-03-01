## Vespa🐝: Video Diffusion State Space Models

This repo contains PyTorch model definitions, pre-trained weights and training/sampling code for our paper video diffusion state space models. 


### 1. Environments

- Python 3.10
  - `conda create -n your_env_name python=3.10`

- Requirements file
  - `pip install -r requirements.txt`

- Install ``causal_conv1d`` and ``mamba``
  - `pip install -e causal_conv1d`
  - `pip install -e mamba`

### 2. Training 

We provide a training script for VeSpa in [`train.py`](train.py). This script can be used to train video diffusion state space models.

To launch DiS-M/2 (64x64) in the raw space training with `N` GPUs on one node:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py \
--model VeSpa-M/2 \
--model-type video \
--dataset-type ucf \
--data-path  /path/to/datat \
--anna-path /path/to/annate \
--image-size 64 \
--lr 1e-4 \
```
