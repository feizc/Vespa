import torch 
import argparse
import os 
import torchvision
import math 
import random
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler 
from torchvision import transforms
from glob import glob
from collections import OrderedDict
from copy import deepcopy
from PIL import Image
from tqdm import tqdm 
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL 

from models_vespa import VeSpa_models 
from diffusion import create_diffusion
from tools.dataset import MSCOCODataset  
from clip import FrozenCLIPEmbedder

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag



def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def main(args): 
    # Setup DDP
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True) 
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{model_string_name}-{args.dataset_type}-{args.image_size}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    if args.latent_space == True: 
        model = VeSpa_models[args.model](
            img_size=args.image_size // 8,
            channels=4,
        ) 
    else:
        model = VeSpa_models[args.model](
            img_size=args.image_size,
            channels=3,
        ) 

    if args.resume is not None:
        print('resume model')
        checkponit = torch.load(args.resume, map_location=lambda storage, loc: storage)['ema'] 
        
        model.load_state_dict(checkponit) 

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    model = DDP(model.to(device), device_ids=[rank]) 
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule 

    if args.latent_space == True: 
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)


    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    print('lr: ', args.lr)

    # Setup data
    

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    clip = FrozenCLIPEmbedder(
        path='/TrainData/Multimodal/michael.fan/ckpts/sdxl-turbo',
        device=device,
    )
    clip.eval()
    clip = clip.to(device)
    

    if args.dataset_type == "mscoco": 
        dataset = MSCOCODataset(
            root=args.data_path,
            annFile=args.anna_path, 
            transform=transform,
        )
    else:
        pass

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )

    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    update_ema(ema, model.module, decay=0) 
    model.train() 
    ema.eval()

    # Variables for monitoring/logging purposes
    train_steps = 0
    log_steps = 0
    running_loss = 0

    for epoch in range(args.epochs): 
        sampler.set_epoch(epoch) 
        running_loss = 0
        train_steps = 0
        with tqdm(enumerate(loader), total=len(loader)) as tq:
            for data_iter_step, samples in tq: 
                # we use a per iteration (instead of per epoch) lr scheduler
                if data_iter_step % args.accum_iter == 0:
                    adjust_learning_rate(opt, data_iter_step / len(loader) + epoch, args)
                
                x = samples[0].to(device) 
                y = samples[1]
                
                with torch.no_grad():
                    context = clip.encode(y)

                if args.latent_space == True: 
                    # Map input images to latent space + normalize latents:
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)

                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device) 
                
                model_kwargs = dict(context=context) 
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                loss_value = loss.item() 

                if not math.isfinite(loss_value): 
                    continue
                
                if (data_iter_step + 1) % args.accum_iter == 0:
                    opt.zero_grad()
                
                loss.backward()
                opt.step()
                update_ema(ema, model.module)

                running_loss += loss_value
                log_steps += 1
                train_steps += 1

                tq.set_description('Epoch %i' % epoch) 
                tq.set_postfix(loss=running_loss / train_steps)
                # Save DiT checkpoint:
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    if rank == 0:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args
                        }
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path) 
                    dist.barrier()
                # break 
                """
                # debug for ignored parameters 
                for name, param in model.named_parameters():
                    if param.grad is None:
                        print(name)
                """
                
                # eval 
                if train_steps % args.eval_steps == 0: 
                    dist.barrier()
                    if rank == 0:
                        diffusion_eval = create_diffusion(str(10)) 
                        n = context.size(0)

                        if args.latent_space == True: 
                            z = torch.randn(n, 4, args.image_size//8, args.image_size//8, device=device)
                        else:
                            z = torch.randn(n, 3, args.image_size, args.image_size, device=device)
                        
                        # Setup classifier-free guidance:
                        # z = torch.cat([z, z], 0)
                        
                        model_kwargs = dict(context=context,)
                        # Sample images:
                        samples = diffusion_eval.p_sample_loop(
                            ema.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                        )
                        eval_samples, _ = samples.chunk(2, dim=0) 
                        
                        if args.latent_space == True: 
                            eval_samples = vae.decode(eval_samples / 0.18215).sample
                            
                        save_image(eval_samples, os.path.join(experiment_dir, "sample_" + str(train_steps // args.eval_steps) + ".png"), nrow=4, normalize=True, value_range=(-1, 1))
                        
                    dist.barrier()



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--anna-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="/TrainData/Multimodal/zhengcong.fei/vespa/results")
    parser.add_argument("--dataset-type", type=str, choices=['mscoco', 'laion'], default='mscoco')
    parser.add_argument("--resume", type=str, default=None)
    
    parser.add_argument("--model", type=str, choices=list(VeSpa_models.keys()), default="VeSpa-L/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512, 64, 32], default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=420) 

    parser.add_argument('--lr', type=float, default=1e-4,) 
    parser.add_argument('--min_lr', type=float, default=1e-6,)
    parser.add_argument('--warmup_epochs', type=int, default=5,)
    parser.add_argument('--accum_iter', default=1, type=int,) 
    parser.add_argument('--eval_steps', default=1000, type=int,) 

    parser.add_argument('--latent_space', type=bool, default=False,) 
    parser.add_argument('--vae_path', type=str, default='/TrainData/Multimodal/zhengcong.fei/dis/vae') 
    args = parser.parse_args()
    main(args)
