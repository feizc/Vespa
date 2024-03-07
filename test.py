import os 
import torch
import torchvision



def test_timeembedding(): 
    from models_dis import timestep_embedding
    times_steps = torch.randint(1, 100, (1,))
    print(timestep_embedding(times_steps, 1000)) 



def test_cifar10(): 
    data_path = "/TrainData/Multimodal/zhengcong.fei/dis/data"
    cifar10 = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=False
    )
    cifar10_test = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=False
    )
    print(cifar10)
    print(cifar10_test[0])



def test_imagenet1k(): 
    data_path = '/TrainData/Multimodal/public/datasets/ImageNet/train' 
    import torchvision.datasets as datasets
    dataset_train = datasets.ImageFolder(data_path) 
    print(dataset_train[0])



def test_celeba(): 
    from datasets import load_dataset
    data_path = "/TrainData/Multimodal/zhengcong.fei/dis/data/CelebA"
    dataset = load_dataset(data_path) 
    # dataset = dataset['train']
    # dataset = dataset.map(lambda e: e['image'].convert('RGB'), batched=True)
    #print(dataset[0])
    print(dataset['train'][0].keys())
    #print(dataset['train'][0]['image'].convert("RGB"))
    # print(len(dataset['train']))


def test_fid_score(): 
    from tools.fid_score import calculate_fid_given_paths 
    path1 = '/TrainData/Multimodal/zhengcong.fei/dis/results/cond_cifar10_small/his'
    path2 = '/TrainData/Multimodal/zhengcong.fei/dis/results/uncond_cifar10_small/his'
    fid = calculate_fid_given_paths((path1, path2))



def test_vae(): 
    from diffusers.models import AutoencoderKL 
    vae_path = '/TrainData/Multimodal/zhengcong.fei/dis/vae'
    vae = AutoencoderKL.from_pretrained(vae_path)


def test_clip(): 
    from transformers import CLIPTokenizer, CLIPTextModel
    clip_path = '/TrainData/Multimodal/michael.fan/ckpts/sdxl-turbo/text_encoder'
    tokenizer_path = '/TrainData/Multimodal/michael.fan/ckpts/sdxl-turbo/tokenizer'
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    transformer = CLIPTextModel.from_pretrained(clip_path)

    text = ['HighJump']
    batch_encoding = tokenizer(text, truncation=True, max_length=77, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    print(tokens.size())
    tokens = tokenizer.convert_ids_to_tokens(tokens.tolist()[0])
    print(tokens)


def test_t5(): 
    from t5 import T5Embedder 
    t5_path = '/TrainData/Multimodal/michael.fan/ckpts/DeepFloyd/t5-v1_1-xxl' 
    
    t5 = T5Embedder(device='cuda', local_cache=True, cache_dir=t5_path) 

    prompts = ['state space models for video generation', 'mamba model is good'] 
    with torch.no_grad(): 
        caption_embs, emb_masks = t5.get_text_embeddings(prompts)
        caption_embs = caption_embs.float() # [:, None]
        print(caption_embs.size(), emb_masks.size())
    

def test_vespa(): 
    from models_vespa import timestep_embedding, VeSpa_models
    from thop import profile

    for k, v in VeSpa_models.items(): 
        print(k)
        model = v(img_size=32).cuda()
        input_image = torch.randn(1, 3, 32, 32).cuda()
        times_steps = torch.randint(1, 100, (1,)).cuda()
        context = torch.randn(1, 77, 768).cuda()
        flops, _ = profile(model, inputs=(input_image, times_steps, context))
        # out = model(x=input_image, timesteps=times_steps)
        #print(out.size())
        print('FLOPs = ' + str(flops * 2/1000**3) + 'G')
        
        parameters_sum = sum(x.numel() for x in model.parameters())
        print(parameters_sum / 1000000.0, "M")



def test_coco(): 
    from tools.dataset import MSCOCODataset
    annafile = '/TrainData/Multimodal/will.zhang/data/coco2014/annotations/captions_train2014.json'
    root = '/TrainData/Multimodal/will.zhang/data/coco2014/train2014/train2014'
    dataset = MSCOCODataset(root=root, annFile=annafile,)
    print(dataset[0])



def test_mjdataset(): 
    from tools.dataset import MJDataset 
    data_path = '/TrainData/Multimodal/public/datasets_gen/mj580w/cleaned_mj_580w.json' 
    dataset = MJDataset(path=data_path)
    print(dataset[0])


def test_video(): 
    from einops import rearrange
    f = 6 
    frames = torch.randn(4 * f, 64*64, 768) 
    print(frames.size())
    frames = rearrange(frames, "(b f) n d -> (b n) f d", f=f)
    print(frames.size())
    frames = rearrange(frames, "(b n) f d -> (b f) n d", b=4)
    print(frames.size())


def ucf_dataset_create(): 
    data_path = '/TrainData/Multimodal/public/datasets_gen/video_dataset/UCF-101'
    import json 
    file_list_path = os.listdir(data_path) 
    print(file_list_path) 
    
    video_test_list = []
    for file in file_list_path: 
        avi_path_list = os.listdir(os.path.join(data_path, file))
        for avi_path in avi_path_list: 
            video_test_list.append(
                {
                    "video": os.path.join(data_path, file, avi_path),
                    "text": file,
                }
            )
    print(len(video_test_list))
    target_path = '/TrainData/Multimodal/zhengcong.fei/vespa/data/ucf.json'
    with open(target_path, 'w') as f: 
        json.dump(video_test_list, f, indent=4)


def test_ucf_dataset(): 
    from tools.dataset import UCFDataset 
    data_path =  '/TrainData/Multimodal/zhengcong.fei/vespa/data/ucf.json'
    dataset = UCFDataset(data_path, is_image=False)
    print(dataset[1][0].size())
    # ([8, 3, 64, 64])



def test_video_vespa(): 
    from models_vespa import VeSpa_video_models
    model = VeSpa_video_models['VeSpa-M/2'](
        img_size=64,
        channels=32,
        enable_temporal_layers=True,
    ) 
    print(model)
    parameters_sum = sum(x.numel() for x in model.parameters())
    print(parameters_sum / 1000000.0, "M")



def face_create(): 
    import json 
    data_path = '/TrainData/Multimodal/public/datasets_gen/video_dataset/face/training_AV/RAVDESS/train'
    file_list_path = os.listdir(data_path)
    print(file_list_path)
    video_test_list = []

    for file in file_list_path: 
        video_test_list.append(
            {
                "video": os.path.join(data_path, file),
                "text": file[:-4].split('_')[2].lower(),
            }
        )
    # print(video_test_list)
    
    print(len(video_test_list))
    target_path = '/TrainData/Multimodal/zhengcong.fei/vespa/data/face.json'
    with open(target_path, 'w') as f: 
        json.dump(video_test_list, f, indent=4)
    from tools.dataset import FaceDataset 
    dataset = FaceDataset(target_path, is_image=False)
    print(dataset[1][0].size())

# face_create()
# test_video_vespa()
# test_ucf_dataset()
# test_video()
# test_mjdataset()
# test_clip() 
# test_vespa() 
# test_coco()
# test_cifar10()
# test_imagenet1k()
# test_celeba()
# test_fid_score()
# test_vae() 
test_t5()