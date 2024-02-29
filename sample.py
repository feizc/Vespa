import torch 
import argparse 
import random 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

from diffusion import create_diffusion
from models_vespa import VeSpa_models 
from clip import FrozenCLIPEmbedder


def main(args):
    print("Sample images from a trained vespa model.")
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False) 
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    checkponit = torch.load(args.ckpt, map_location=lambda storage, loc: storage)['ema'] 
    model.load_state_dict(checkponit) 
    model = model.to(device)
    model.eval() 

    diffusion = create_diffusion(str(args.num_sampling_steps)) 
    if args.latent_space == True: 
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)

    clip = FrozenCLIPEmbedder(
        path='/TrainData/Multimodal/michael.fan/ckpts/sdxl-turbo',
        device=device,
    )
    clip.eval()
    clip = clip.to(device)
    
    n = 4
    text = ['a cute cat in grass', 'water with ocean', 'a cute cat in grass', 'water with ocean',]
    
    if args.latent_space == True: 
        z = torch.randn(n, 4, args.image_size//8, args.image_size//8, device=device)
    else:
        z = torch.randn(n, 3, args.image_size, args.image_size, device=device)
    
    # Setup classifier-free guidance:
    # z = torch.cat([z, z], 0)
    
    with torch.no_grad(): 
        context = clip.encode(text)

    model_kwargs = dict(context=context,)
    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    eval_samples, _ = samples.chunk(2, dim=0) 
    
    if args.latent_space == True: 
        eval_samples = vae.decode(eval_samples / 0.18215).sample
    
    save_image(eval_samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))



if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model", type=str, choices=list(VeSpa_models.keys()), default="VeSpa-H/2")
    parser.add_argument("--image-size", type=int, choices=[32, 64, 256, 512], default=256) 
    parser.add_argument("--cfg-scale", type=float, default=1.5) 
    parser.add_argument("--num-sampling-steps", type=int, default=250) 
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default="/TrainData/Multimodal/zhengcong.fei/vespa/results/VeSpa-H-2-mj-256/checkpoints/0120000.pt",) 
    parser.add_argument('--latent_space', type=bool, default=True,) 
    parser.add_argument('--vae_path', type=str, default='/TrainData/Multimodal/zhengcong.fei/dis/vae') 
    args = parser.parse_args()

    main(args)