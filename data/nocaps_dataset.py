import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

class nocaps_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):   
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/nocaps_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/nocaps_test.json'}
        filenames = {'val':'nocaps_val.json','test':'nocaps_test.json'}
        
        # Prefer local annotation files if present; only download when missing.
        ann_path = os.path.join(ann_root, filenames[split])
        if not os.path.exists(ann_path):
            try:
                download_url(urls[split], ann_root)
            except Exception as e:  # pragma: no cover - offline or network failure
                # Fallback to local-only if provided by the user (e.g., Kaggle Data tab)
                pass
        
        if not os.path.exists(ann_path):
            # If still missing, raise a clear error
            raise FileNotFoundError(f"Missing NoCaps annotation file: {ann_path}")
        
        self.annotation = json.load(open(ann_path,'r'))
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):  
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        return image, int(ann['img_id'])    
