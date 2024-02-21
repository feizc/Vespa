import torch 
import os 
import einops
import numpy as np
from PIL import Image
from random import choice 

from datasets import load_dataset
from torch.utils.data import Dataset
from datasets import load_dataset


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




