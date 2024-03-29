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
from tools.dataset import MSCOCODataset, MJDataset, UCFDataset, FaceDataset, wds_process, TagImageNetDataset
from clip import FrozenCLIPEmbedder 
from t5 import T5Embedder 



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
        experiment_dir = f"{args.results_dir}/{model_string_name}-{args.dataset_type}-{args.model_type}-{args.image_only}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
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
        print('resume model')
        checkponit = torch.load(args.resume, map_location=lambda storage, loc: storage)['ema'] 
        
        if args.image_only == False: 
            model.load_state_dict(checkponit, strict=False) 
        else:
            model.load_state_dict(checkponit) 


    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    if args.image_only == False: 
        model.requires_grad_(False)
        temporal_params = model.temporal_parameters()
        for p in temporal_params: 
            # zero out for temporal
            # p.detach().zero_() 
            p.data.fill_(0) 
            p.requires_grad_(True) 

    model = DDP(model.to(device), device_ids=[rank]) 
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule 

    if args.latent_space == True: 
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, )
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
            device=device,
        )
        text_encoder.eval()
        text_encoder = text_encoder.to(device)
    elif args.text_encoder_type == 't5':
        t5_path = '/maindata/data/shared/multimodal/zhengcong.fei/ckpts/DeepFloyd/t5-v1_1-xxl' 
        text_encoder = T5Embedder(device='cuda', local_cache=True, cache_dir=t5_path) 
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
    elif args.dataset_type == 'imagenet':
        dataset = TagImageNetDataset(
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
        import webdataset as wds 
        tar_list = os.listdir(args.anna_path)
        urls = [os.path.join(args.anna_path, f) for f in tar_list]
        process = wds_process(transform)
        dataset = wds.DataPipeline(
            wds.SimpleShardList(urls),
            # at this point we have an iterator over all the shards
            wds.shuffle(len(urls)),
            # add wds.split_by_node here if you are using multiple nodes
            wds.split_by_worker,
            wds.split_by_node,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(),
            # this shuffles the samples in memory
            wds.shuffle(1000),
            # this decodes the images and json
            wds.map(process),
            wds.shuffle(1000),
            wds.batched(int(args.global_batch_size // dist.get_world_size()),)
        )
    else:
        pass
    
    if args.dataset_type == 'wds': 
        sampler = None
        loader = dataset
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

    update_ema(ema, model.module, decay=0) 
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
        try: 
            length = len(loader)
        except:
            length = 5800000
        
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
                
                loss.backward()
            
                opt.step()
                update_ema(ema, model.module)

                running_loss += loss_value
                log_steps += 1
                train_steps += 1

                tq.set_description('Epoch %i' % epoch) 
                tq.set_postfix(loss=running_loss / (data_iter_step + 1))
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
    parser.add_argument("--results-dir", type=str, default="/maindata/data/shared/multimodal/zhengcong.fei/code/vespa/results")
    parser.add_argument("--dataset-type", type=str, choices=['mscoco', 'ucf', 'mj', 'face', 'wds', 'imagenet'], default='mscoco')
    parser.add_argument("--image-only", type=bool, default=False)
    parser.add_argument("--text_encoder_type", type=str, choices=['clip', 't5'], default='clip')
    parser.add_argument("--resume", type=str, default=None)
    
    parser.add_argument("--model", type=str, default="VeSpa-L/2")
    parser.add_argument("--model-type", type=str, default="image")
    parser.add_argument("--image-size", type=int, choices=[256, 512, 64, 32], default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--ckpt-every", type=int, default=3000)
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
