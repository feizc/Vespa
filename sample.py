import torch 
import argparse 
import random 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

from diffusion import create_diffusion
from models_vespa import VeSpa_image_models, VeSpa_video_models 
from clip import FrozenCLIPEmbedder
from t5 import T5Embedder 


def main(args):
    print("Sample images from a trained vespa model.")
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False) 
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    checkponit = torch.load(args.ckpt, map_location=lambda storage, loc: storage)['ema'] 
    model.load_state_dict(checkponit) 
    model = model.to(device)
    model.eval() 

    diffusion = create_diffusion(str(args.num_sampling_steps)) 
    if args.latent_space == True: 
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)

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
    
    n = 16
    y = ['tiger cat',] * n
    # text = ['Skiing',] * n
    
    if args.latent_space == True: 
        z = torch.randn(n, 4, args.image_size//8, args.image_size//8, device=device)
    else:
        z = torch.randn(n, 3, args.image_size, args.image_size, device=device)
    
    # Setup classifier-free guidance:
    # z = torch.cat([z, z], 0)
    
    with torch.no_grad(): 
        if args.text_encoder_type == 'clip': 
            context = text_encoder.encode(y)
        else:
            context, _ = text_encoder.get_text_embeddings(y)
            context = context.float() 

    if args.image_only == True: 
        model_kwargs = dict(context=context,)
    else:
        model_kwargs = dict(context=context, f=8)
    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    eval_samples, _ = samples.chunk(2, dim=0) 
    # eval_samples = samples

    if args.latent_space == True: 
        eval_samples = vae.decode(eval_samples / 0.18215).sample
    
    save_image(eval_samples, "sample.png", nrow=8, normalize=True, value_range=(-1, 1))



if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model", type=str, default="VeSpa-H/2")
    parser.add_argument("--model-type", type=str, default="image")
    parser.add_argument("--text_encoder_type", type=str, choices=['clip', 't5'], default='t5')
    parser.add_argument("--image-size", type=int, choices=[32, 64, 256, 512], default=256) 
    parser.add_argument("--image-only", type=bool, default=True)
    parser.add_argument("--cfg-scale", type=float, default=1.5) 
    parser.add_argument("--num-sampling-steps", type=int, default=250) 
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default="/maindata/data/shared/multimodal/zhengcong.fei/code/vespa/results/VeSpa-H-2-imagenet-image-True/checkpoints/0033000.pt",) 
    # parser.add_argument("--ckpt", type=str, default="/TrainData/Multimodal/zhengcong.fei/vespa/results/VeSpa-M-2-face-video-False/checkpoints/0024000.pt",) 
    # parser.add_argument("--ckpt", type=str, default="/TrainData/Multimodal/zhengcong.fei/vespa/results/VeSpa-M-2-ucf-video-False/checkpoints/0024000.pt",) 
    parser.add_argument('--latent_space', type=bool, default=True,) 
    parser.add_argument('--vae_path', type=str, default='/maindata/data/shared/multimodal/zhengcong.fei/ckpts/playground/vae') 
    args = parser.parse_args()

    main(args)
