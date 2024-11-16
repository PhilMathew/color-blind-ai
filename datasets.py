import cv2
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import CelebA, LFWPairs
from torchvision import transforms
from typing import *


IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def convert_to_greyscale(img: Tensor):
    img = img.permute(1, 2, 0)
    img = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2GRAY)
    
    return Tensor(img).unsqueeze(0)


class CelebADataset(Dataset):
    def __init__(self, data_path: str, split: str, img_size: Tuple[int, int] = (128, 128)):
        self.data_path = data_path
        self.split = split
        
        match self.split: # figure out which transform to use
            case 'train':
                self.transform = transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.Resize(img_size),
                        transforms.ToTensor(), 
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                    ]
                )
            case 'valid':
                self.transform = transforms.Compose(
                    [
                        transforms.Resize(img_size),
                        transforms.ToTensor(), 
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                    ]
                )
            case 'test':
                self.transform = transforms.Compose(
                    [
                        transforms.Resize(img_size),
                        transforms.ToTensor(), 
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                    ]
                )
            case _:
                raise ValueError(f"Invalid split: {split}")
            
        self.dataset = CelebA(
            data_path, 
            split=split, 
            download=True, 
            target_type='identity',
            transform=self.transform
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        img, label = self.dataset[i]
        grey_img = convert_to_greyscale(img)
        
        return img, grey_img, label


class LFWPairsDataset(Dataset):
    def __init__(self, data_path: str, img_size: Tuple[int, int] = (128, 128)):
        self.data_path = data_path
        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(), 
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        self.dataset = LFWPairs(
            data_path,
            download=True,
            transform=self.transform
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        img1, img2, label = self.dataset[i]
        
        return img1, img2, label
