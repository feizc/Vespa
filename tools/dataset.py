import torch 
import os 
import einops 
import random 
import numpy as np
import json 
from PIL import Image
from random import choice 

from datasets import load_dataset
from torch.utils.data import Dataset
from datasets import load_dataset
import torchvision.transforms as transforms
from decord import VideoReader



class CelebADataset(Dataset): 
    def __init__(self, data_path, transform): 
        data = load_dataset(data_path) 
        self.data = data['train']
        self.transform = transform 
    
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, index): 
        image = self.data[index]['image'].convert("RGB") 
        return self.transform(image), torch.tensor(index).long()



def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)


class MSCOCODataset(Dataset): 
    def __init__(self, root, annFile, transform, ): 
        from pycocotools.coco import COCO
        self.root = root

        self.coco = COCO(annFile) 
        self.keys = list(sorted(self.coco.imgs.keys())) 
        self.transform = transform 
    
    def _load_image(self, key: int):
        path = self.coco.loadImgs(key)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, key: int):
        return self.coco.loadAnns(self.coco.getAnnIds(key))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        image = self._load_image(key)
        image = self.transform(image)

        anns = self._load_target(key)
        target = []
        for ann in anns:
            target.append(ann['caption'])
        
        return image, choice(target)



class MJDataset(Dataset): 
    def __init__(self, path, transform): 
        with open(path, 'r') as f: 
            self.data = json.load(f) 
        self.key_list = [key for key in self.data.keys()]
        self.transform = transform 

    def __len__(self): 
        return len(self.key_list)

    def __getitem__(self, index): 
        img_path = self.key_list[index]
        img = Image.open(img_path).convert("RGB") 
        img = self.transform(img)
        txt = self.data[img_path]['caption'] 
        return img, txt 




class UCFDataset(Dataset): 
    def __init__(
        self, 
        data_path,
        sample_size=64,
        sample_stride=4,
        sample_n_frames=8,
        is_image=True,
    ):
        with open(data_path, 'r') as f: 
            self.dataset = json.load(f) 
        
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames 

        self.is_image = is_image 
        self.length = len(self.dataset)

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def get_batch(self, idx): 
        name = self.dataset[idx]['text']
        video_reader = VideoReader(self.dataset[idx]['video']) 
        video_length = len(video_reader)
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]
        
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]
        
        return pixel_values, name

    
    def __len__(self): 
        return self.length 
    
    def __getitem__(self, idx): 
        pixel_values, name = self.get_batch(idx) 
        pixel_values = self.pixel_transforms(pixel_values)
        return pixel_values, name 



class FaceDataset(Dataset): 
    def __init__(
        self, 
        data_path,
        sample_size=64,
        sample_stride=4,
        sample_n_frames=8,
        is_image=True,
    ):
        with open(data_path, 'r') as f: 
            self.dataset = json.load(f) 
        
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames 

        self.is_image = is_image 
        self.length = len(self.dataset)

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def get_batch(self, idx): 
        name = self.dataset[idx]['text']
        video_reader = VideoReader(self.dataset[idx]['video']) 
        video_length = len(video_reader)
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]
        
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]
        
        return pixel_values, name

    
    def __len__(self): 
        return self.length 
    
    def __getitem__(self, idx): 
        pixel_values, name = self.get_batch(idx) 
        pixel_values = self.pixel_transforms(pixel_values)
        return pixel_values, name 