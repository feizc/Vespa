## Vespa🐝: Video Diffusion State Space Models

This repo contains PyTorch model definitions, pre-trained weights and training/sampling code for our paper video diffusion state space models. 
Our model use clip/t5 as text encoder and mamba-based diffusion model. 
Its distinctive advantage lies in ites reduced spatial complexity, which renders it exceptionally adept at processing long videos or high-resolution images, eliminating the necessity for window operations. 


![sad](https://github.com/feizc/Vespa/assets/37614046/5bcd0cba-9cb0-4cba-ab36-801539722709)


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
--lr 1e-4
```


### 3. Evaluation

We include a [`sample.py`](sample.py) script which samples images from a DiS model. Besides, we support other metrics evaluation, e.g., FLOPS and model parameters, in [`test.py`](test.py) script. 

```bash
python sample.py \
--model VeSpa-M/2 \
--ckpt /path/to/model \
--image-size 64 \
--prompt sad 
```

### 4. BibTeX

```bibtex
@article{FeiVespa2024,
  title={Video Diffusion State Space Models},
  author={Zhengcong Fei, Mingyuan Fan, Changqian Yu, Jusnshi Huang},
  year={2024},
  journal={arXiv preprint},
}
```


### 5. Acknowledgments

The codebase is based on the awesome [DiS](https://github.com/feizc/DiS), [DiT](https://github.com/facebookresearch/DiT), [mamba](https://github.com/state-spaces/mamba), [U-ViT](https://github.com/baofff/U-ViT), and [Vim](https://github.com/hustvl/Vim) repos. 







