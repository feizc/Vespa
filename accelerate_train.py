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

from einops import rearrange, repeat
from models_vespa import VeSpa_image_models, VeSpa_video_models 
from diffusion import create_diffusion
from tools.dataset import MSCOCODataset, MJDataset, UCFDataset, FaceDataset, wds_process
from tools.webdataset import Webdataset_Vespa
from clip import FrozenCLIPEmbedder 
from t5 import T5Embedder 
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device

@torch.no_grad()
def update_ema_accelerate(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    adjusted_model_params = {}
    for name, param in model_params.items():
        adjusted_name = name
        
        if any(name not in ema_params and f'module.module.{name}' in ema_params for name in model_params):
            adjusted_name = f'module.module.{name}'
        adjusted_model_params[adjusted_name] = param

    for name, param in adjusted_model_params.items():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
        else:
            print(f"Warning: {name} is not found in EMA model.")

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


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

    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = accelerator.state.process_index
    world_size = accelerator.state.num_processes

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    accelerator.print(f"Starting seed={seed}, world_size={world_size}.")

    # Setup an experiment folder
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)
        # experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{model_string_name}-{args.dataset_type}-{args.model_type}-{args.image_only}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    if args.latent_space == True: 
        img_size = args.image_size // 8
        channels = 4
    else: 
        img_size=args.image_size
        channels = 3 
    
    if args.text_encoder_type == 'clip': 
        num_clip_token = 77 
        clip_dim = 768
    else:
        num_clip_token = 120 
        clip_dim = 4096

    if args.model_type == 'image': 
        model = VeSpa_image_models[args.model](
            img_size=img_size,
            channels=channels,
            num_clip_token=num_clip_token,
            clip_dim=clip_dim,
        ) 
    else:
        model = VeSpa_video_models[args.model](
            img_size=img_size,
            channels=channels,
            enable_temporal_layers= not args.image_only, 
            num_clip_token=num_clip_token,
            clip_dim=clip_dim,
        ) 

        if args.resume is not None:
            accelerator.print('resume model')
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)['ema']

        if args.image_only == False:
            model.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint)

    if args.image_only == False:
        print(model)

        model.requires_grad_(False)
        temporal_params = model.temporal_parameters()
        for p in temporal_params:
            # zero out for temporal
            # p.detach().zero_()
            # p.data.fill_(0)
            p.requires_grad_(True)

    
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule 

    if args.latent_space == True: 
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, )
    model, opt = accelerator.prepare(model,opt)
    ema = deepcopy(model)
    ema = accelerator.prepare(ema)
    requires_grad(ema, False)
    print('lr: ', args.lr)

    # Setup data

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    if args.text_encoder_type == 'clip':
        text_encoder = FrozenCLIPEmbedder(
            path='/maindata/data/shared/multimodal/zhengcong.fei/ckpts/playground',
            device=accelerator.device,
        )
        text_encoder.eval()
        text_encoder = accelerator.prepare(text_encoder)

    elif args.text_encoder_type == 't5':
        t5_path = '/maindata/data/shared/multimodal/zhengcong.fei/ckpts/DeepFloyd/t5-v1_1-xxl' 
        text_encoder = T5Embedder(device=accelerator.device, local_cache=True, cache_dir=t5_path)
        text_encoder = accelerator.prepare(text_encoder)
    else:
        pass
    

    if args.dataset_type == "mscoco": 
        dataset = MSCOCODataset(
            root=args.data_path,
            annFile=args.anna_path, 
            transform=transform,
        )
    elif args.dataset_type == "mj": 
        dataset = MJDataset(
            path=args.anna_path, 
            transform=transform,
        )
    elif args.dataset_type == "ucf":
        dataset = UCFDataset(
            data_path=args.anna_path,
            sample_size=args.image_size,
            is_image=args.image_only,
        ) 
    elif args.dataset_type == "face":
        dataset = FaceDataset(
            data_path=args.anna_path,
            sample_size=args.image_size,
            is_image=args.image_only,
        ) 
    elif args.dataset_type == 'wds':
        wds_vespa = Webdataset_Vespa(args, 
            transform=transform, 
            num_train_examples=5800000, 
            world_size=world_size,
            per_gpu_batch_size=args.global_batch_size//world_size, 
            global_batch_size=args.global_batch_size, 
            num_workers=os.cpu_count())
    else:
        pass
    
    if args.dataset_type == 'wds': 
        sampler = None
        loader = wds_vespa.train_dataset
    else:
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

    try: 
        length = len(loader)
    except:
        length = wds_vespa.train_dataloader.num_batches
        
    loader = accelerator.prepare(loader)
    
    update_ema_accelerate(ema, model.module, decay=0) 
    model.train() 
    ema.eval()

    # Variables for monitoring/logging purposes
    train_steps = 0
    log_steps = 0
    running_loss = 0

    for epoch in range(args.epochs): 
        if sampler is not None:
            sampler.set_epoch(epoch) 

        running_loss = 0
        with tqdm(enumerate(loader), total=length) as tq:
            for data_iter_step, samples in tq: 
                # we use a per iteration (instead of per epoch) lr scheduler
                if data_iter_step % args.accum_iter == 0:
                    adjust_learning_rate(opt, data_iter_step / length + epoch, args)
                
                x = samples[0].to(device) 
                y = samples[1]
                b = x.size(0)
                with torch.no_grad(): 
                    if args.text_encoder_type == 'clip': 
                        context = text_encoder.encode(y)
                    else:
                        context, _ = text_encoder.get_text_embeddings(y)
                        context = context.float() 

                f = 0
                if args.image_only == False: 
                    f = x.size(1) 
                    x = rearrange(x, "b f c h w -> (b f) c h w") 
                
                if args.latent_space == True: 
                    # Map input images to latent space + normalize latents:
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)

                t = torch.randint(0, diffusion.num_timesteps, (b,), device=device) 
                if args.image_only == False: 
                    t = repeat(t, 'b -> (b f)', f=f)
                    context = repeat(context, 'b l d -> (b f) l d', f=f)

                if args.model_type == 'image': 
                    model_kwargs = dict(context=context, ) 
                else:
                    model_kwargs = dict(context=context, f=f) 
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                loss_value = loss.item() 

                if not math.isfinite(loss_value): 
                    continue
                
                if (data_iter_step + 1) % args.accum_iter == 0:
                    opt.zero_grad()
                
                accelerator.backward(loss)
            
                opt.step()
                update_ema_accelerate(ema, model.module)
                running_loss += loss_value
                log_steps += 1
                train_steps += 1

                tq.set_description('Epoch %i' % epoch) 
                tq.set_postfix(loss=running_loss / (data_iter_step + 1))
                # Save DiT checkpoint:
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args
                        }
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        print(f"Saving DiT checkpoint:{checkpoint_path}")
                        torch.save(checkpoint, checkpoint_path) 
                # break 
                """
                # debug for ignored parameters 
                for name, param in model.named_parameters():
                    if param.grad is None:
                        print(name)
                """
                
                # eval 
                if train_steps % args.eval_steps == 0:
                    continue # avoid oom
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process == 0: 
                        print(f'Evaluating {train_steps} steps performance')
                        diffusion_eval = create_diffusion(str(250)) 
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
                            ema.module.module.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                        )
                        eval_samples, _ = samples.chunk(2, dim=0) 
                        
                        if args.latent_space == True: 
                            eval_samples = vae.decode(eval_samples / 0.18215).sample
                            
                        save_image(eval_samples, os.path.join(experiment_dir, "sample_" + str(train_steps // args.eval_steps) + ".png"), nrow=4, normalize=True, value_range=(-1, 1))
                            
                    accelerator.wait_for_everyone()



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--anna-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="/maindata/data/shared/multimodal/yujun.liu/codes/vespa/results")
    parser.add_argument("--dataset-type", type=str, choices=['mscoco', 'ucf', 'mj', 'face', 'wds'], default='mscoco')
    parser.add_argument("--image-only", type=bool, default=False)
    parser.add_argument("--text_encoder_type", type=str, choices=['clip', 't5'], default='clip')
    parser.add_argument("--resume", type=str, default=None)
    
    parser.add_argument("--model", type=str, default="VeSpa-L/2")
    parser.add_argument("--model-type", type=str, default="image")
    parser.add_argument("--image-size", type=int, choices=[256, 512, 64, 32], default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--ckpt-every", type=int, default=30)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=420) 

    parser.add_argument('--lr', type=float, default=1e-4,) 
    parser.add_argument('--min_lr', type=float, default=1e-6,)
    parser.add_argument('--warmup_epochs', type=int, default=5,)
    parser.add_argument('--accum_iter', default=1, type=int,) 
    parser.add_argument('--eval_steps', default=1000, type=int,) 

    parser.add_argument('--latent_space', type=bool, default=False,) 
    parser.add_argument('--vae_path', type=str, default='/maindata/data/shared/multimodal/zhengcong.fei/ckpts/playground/vae') 
    
    args = parser.parse_args()
    main(args)
