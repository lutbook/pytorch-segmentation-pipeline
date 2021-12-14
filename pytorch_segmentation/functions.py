import os, copy, random
import pandas as pd
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    """Custom Dataset"""
    def __init__(self, phase, dataset_dir, csv_file, image_transform=None, target_transform=None):#, sample_resize=None):
        """
        Args:
            phase (string):
            csv_file (string):
            dataset_dir (string):
            transform (callable, optional):
        """
        self.phase = phase
        self.dataset_dir = dataset_dir
        self.dataset_frame = pd.read_csv(csv_file)
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.images = []
        # self.sample_resize = sample_resize
        self.class_values = self.dataset_frame.name.values

        img_dir = os.path.join(self.dataset_dir, phase, 'images')

        try:
            os.remove(os.path.join(img_dir, '.DS_Store'))
        except:
            pass

        for file_name in os.listdir(img_dir):
            if file_name[0] == '.':
                continue
            self.images.append(os.path.join(img_dir, file_name))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = Image.open(self.images[idx]).convert('RGB')
        target = Image.open(self.images[idx].replace('images', 'labels').replace('.png', '_labelIds.png'))

        # if self.sample_resize:
        #     image = image.resize((self.sample_resize, self.sample_resize))
        #     target = target.resize((self.sample_resize, self.sample_resize), Image.NEAREST)
       
        seed = np.random.randint(0, 2**16)
        if self.image_transform is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.image_transform(image)
        if self.target_transform is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            target = self.target_transform(target)

        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

def imshow(inp, num_classes, colors, inf_class_idx, mode=None):
    """ Imshow for Tensor. """
    inp = inp.numpy()
    if mode is not None:
        pred = np.zeros((3, inp.shape[0], inp.shape[1]), dtype=np.uint8)
        if inf_class_idx == None:
            for i in range(num_classes):
                mask = inp == i
                pred[0][mask] = colors[i][0]
                pred[1][mask] = colors[i][1] 
                pred[2][mask] = colors[i][2] 
        else:
            mask = inp == inf_class_idx
            pred[0][mask] = colors[inf_class_idx][0]
            pred[1][mask] = colors[inf_class_idx][1] 
            pred[2][mask] = colors[inf_class_idx][2] 

        result = Image.fromarray(pred.transpose(1, 2, 0), 'RGB')
        return result
    else:
        result = inp.transpose((1, 2, 0))
        result = np.clip(result, 0, 1)
        plt.imshow(result)

class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True
    
    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01

def inter_and_union(pred, mask, num_class):
    pred = np.asarray(pred, dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=np.uint8).copy()

    # 255 -> 0
    pred += 1
    mask += 1
    pred = pred * (mask > 0)

    inter = pred * (pred == mask)
    (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class))
    (area_pred, _) = np.histogram(pred, bins=num_class, range=(1, num_class))
    (area_mask, _) = np.histogram(mask, bins=num_class, range=(1, num_class))
    area_union = area_pred + area_mask - area_inter

    return (area_inter, area_union)