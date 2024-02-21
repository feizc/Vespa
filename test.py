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

    text = ['vespa is all you need', '']
    batch_encoding = tokenizer(text, truncation=True, max_length=77, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    print(tokens)



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


test_clip() 
# test_vespa() 
# test_coco()
# test_cifar10()
# test_imagenet1k()
# test_celeba()
# test_fid_score()
# test_vae()