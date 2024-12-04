import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
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

    
class DemographicGroupedDataset(Dataset):
    def __init__(self, annotation_file: str, threshold: float = 1.1, img_size=(112,112)):
        self.annotation_file = annotation_file
        self.threshold = threshold
        self.img_size = img_size
        self.grouped_data = self._process_annotations()
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        
    def _decompose_annotation(self, image_name):
        parts = image_name.split('/')
        task = parts[0].split('_')[-1]  
        seed = parts[1].split('_')[1]  
        attributes = parts[-1].split('_')
        race = attributes[1]  
        gender = attributes[3]  
        num_with_extension = attributes[-1]  
        num = '.'.join(num_with_extension.split('.')[:-1])  
        
        return {
            'task': task,
            'seed': seed,
            'race': race,
            'gender': gender,
            'num': num,
        }


    def _construct_image_path(self, image_name):
        components = self._decompose_annotation(image_name)
        task = components['task']
        seed = components['seed']
        race = components['race']
        gender = components['gender']
        num = components['num']

        # Task-specific suffix and conventions
        if task in ['age', 'lighting', 'smiling']:
            suffix = f"{race}_{gender}_{task}_{num}_o2_rm_bg.png"
        elif task == 'pose':
            suffix = f"{race}_{gender}_y_{num}_o2_rm_bg.png"
        else:
            raise ValueError(f"Unsupported task: {task}")

        # Construct the full path
        path = os.path.join(
            "syndata",
            f"final_picked_{task}",
            f"seed_{seed}",
            suffix
        )

        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        return path
    
    def _load_image(self, path: str):
        image = Image.open(path).convert('RGB')  
        return self.transform(image)

    def _process_annotations(self):
        df = pd.read_csv(self.annotation_file)
        annotator_columns = [col for col in df.columns if col.startswith("A") and "_SCORE" in col]
        def trimmed_mean(row):
            sorted_scores = sorted(row[annotator_columns])
            trimmed_scores = sorted_scores[2:-2]  # Drop 2 lowest and 2 highest, following same processing as the casualface paper
            return sum(trimmed_scores) / len(trimmed_scores)  

        df['average_score'] = df.apply(trimmed_mean, axis=1)
        df['label'] = (df['average_score'] < self.threshold).astype(int)

        def extract_demographic_group(row):
            com1 = self._decompose_annotation(row['IMAGE_NAME1'])
            com2 = self._decompose_annotation(row['IMAGE_NAME2'])
            g1 = com1['gender']
            r1 = com1['race']
            g2 = com2['gender']
            r2 = com2['race']
            
            if r1 == r2 and g1 == g2:
                return f"{r1}_{g1}" 
            return None

        df['demographic_group'] = df.apply(extract_demographic_group, axis=1)
        df = df.dropna(subset=['demographic_group'])

        grouped_data = defaultdict(list)
        for _, row in df.iterrows():
            demographic_group = row['demographic_group']
            image_1 = row['IMAGE_NAME1']
            image_2 = row['IMAGE_NAME2']
            label = row['label']

            formatted_element = (image_1, image_2, label)
            grouped_data[demographic_group].append(formatted_element)

        return grouped_data

    def __len__(self):
        """
        Returns the total number of groups.
        """
        return len(self.grouped_data)

    def __getitem__(self, idx):
        demographic_groups = list(self.grouped_data.keys())
        group_name = demographic_groups[idx]
        group_data = self.grouped_data[group_name]
        img1_path, img2_path, label = group_data[0]
        img1_path = self._construct_image_path(img1_path)
        img2_path = self._construct_image_path(img2_path)

        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)

        return img1, img2, label, group_name