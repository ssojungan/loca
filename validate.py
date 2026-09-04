#!/usr/bin/env python3

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import defaultdict
import argparse
from tqdm import tqdm
import numpy as np

def load_mask_config(config_path):
    """Load mask configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_imagenet_a_config():
    """Load ImageNet-A configuration"""
    config = load_mask_config("configs/imagenet-a.json")
    thousand_k_to_200 = {int(k): v for k, v in config["thousand_k_to_200"].items()}
    indices_in_1k = config["indices_in_1k"]
    return thousand_k_to_200, indices_in_1k

def load_imagenet_r_config():
    """Load ImageNet-R configuration"""
    config = load_mask_config("configs/imagenet-r.json")
    all_wnids = config["all_wnids"]
    imagenet_r_wnids = set(config["imagenet_r_wnids"])
    imagenet_r_mask = [wnid in imagenet_r_wnids for wnid in all_wnids]
    return all_wnids, imagenet_r_wnids, imagenet_r_mask

thousand_k_to_200, indices_in_1k = load_imagenet_a_config()
all_wnids, imagenet_r_wnids, imagenet_r_mask = load_imagenet_r_config()

def load_subset_info(info_path):
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    folder_to_imagenet1k_idx = {}
    folder_to_subset_idx = {}
    subset_idx_to_folder = {}
    subset_idx_to_text = {}
    folder_to_text = {}
    imagenet1k_to_subset = {}
    
    for subset_idx, (original_imagenet1k_idx, class_info) in enumerate(info.items()):
        folder = class_info['cls_folder']
        text = class_info['text']
        original_idx = int(original_imagenet1k_idx)
        
        folder_to_imagenet1k_idx[folder] = original_idx
        folder_to_subset_idx[folder] = subset_idx
        subset_idx_to_folder[subset_idx] = folder
        subset_idx_to_text[subset_idx] = text
        folder_to_text[folder] = text
        imagenet1k_to_subset[original_idx] = subset_idx
    
    return (folder_to_imagenet1k_idx, folder_to_subset_idx, 
            subset_idx_to_folder, subset_idx_to_text, 
            folder_to_text, imagenet1k_to_subset)

def get_imagenet_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_dataset(dataset_name, data_dir):
    """Load dataset using torchvision.datasets.ImageFolder"""
    transform = get_imagenet_transform()
    
    if dataset_name in ['imagenet100', 'imagenette']:
        info_path = f"configs/{dataset_name}_info.json"
        return load_subset_dataset(data_dir, info_path)
    else:
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        return dataset


def load_subset_dataset(data_dir, info_path):
    mapping_info = load_subset_info(info_path)
    folder_to_imagenet1k_idx = mapping_info[0]
    
    samples = []
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for class_folder in class_dirs:
        if class_folder not in folder_to_imagenet1k_idx:
            continue
            
        true_imagenet1k_idx = folder_to_imagenet1k_idx[class_folder]
        class_dir = os.path.join(data_dir, class_folder)
        image_files = [f for f in os.listdir(class_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in image_files:
            samples.append((os.path.join(class_dir, image_file), true_imagenet1k_idx))
    
    return samples, mapping_info


class ValImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, true_label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        tensor = self.transform(img)
        return tensor, true_label, path

def validate_model(model, datasets, device='cuda', batch_size=64, num_workers=4):
    model.eval()
    device = next(model.parameters()).device
    
    all_results = {}
    
    for dataset_name, dataset_info in datasets.items():
        if isinstance(dataset_info, tuple):
            samples, mapping_info = dataset_info
            transform = get_imagenet_transform()
            ds = ValImageDataset(samples, transform)
        else:
            ds = dataset_info
            mapping_info = None
        
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, pin_memory=True)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f'Validating {dataset_name}'):
                if isinstance(dataset_info, tuple):
                    # Custom dataset format: (img, label, path)
                    imgs, true_labels, paths = batch
                else:
                    # ImageFolder format: (img, label)
                    imgs, true_labels = batch
                
                imgs = imgs.to(device, non_blocking=True)
                true_labels = true_labels.to(device)
                
                outputs = model(imgs)
                
                if mapping_info is not None:
                    imagenet1k_to_subset = mapping_info[5]
                    
                    target_indices = sorted(list(imagenet1k_to_subset.keys()))
                    target_indices_tensor = torch.tensor(target_indices, device=device, dtype=torch.long)
                    subset_logits = outputs.index_select(1, target_indices_tensor)
                    subset_probs = F.softmax(subset_logits, dim=1)
                    _, predicted_indices = torch.max(subset_probs, dim=1)
                    predicted_labels = target_indices_tensor[predicted_indices]
                elif dataset_name == 'imagenet-r':
                    imagenet_r_mask_tensor = torch.tensor(imagenet_r_mask, device=device, dtype=torch.bool)
                    masked_outputs = outputs[:, imagenet_r_mask_tensor]
                    predicted_labels = torch.argmax(masked_outputs, dim=1)
                elif dataset_name == 'imagenet-a':
                    indices_in_1k_tensor = torch.tensor(indices_in_1k, device=device, dtype=torch.long)
                    masked_outputs = outputs[:, indices_in_1k_tensor]
                    predicted_labels = torch.argmax(masked_outputs, dim=1)
                else:
                    predicted_labels = torch.argmax(outputs, dim=1)
                
                correct += (predicted_labels == true_labels).sum().item()
                total += true_labels.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        
        all_results[dataset_name] = {
            'accuracy': accuracy,
            'total': total,
            'correct': correct,
            'wrong': total - correct
        }
    
    return all_results

def validate_models(model, datasets=None, device='cuda', batch_size=256, num_workers=4):
    DEFAULT_DATASETS = {
        'imagenet-1k': '/home/dataset/imagenet-1k/val',
        #'imagenet-v2': '/home/dataset/imagenet-v2',
        'imagenet-a': '/home/dataset/imagenet-a',
        'imagenet-r': '/home/dataset/imagenet-r',
        'imagenet-sketch': '/home/dataset/imagenet-sketch'
    }

    if datasets is None:
        datasets = DEFAULT_DATASETS
    
    validation_datasets = {}
    for dataset_name, data_dir in datasets.items():
        samples = load_dataset(dataset_name, data_dir)
        validation_datasets[dataset_name] = samples

    results = validate_model(model, validation_datasets, device=device, 
                           batch_size=batch_size, num_workers=num_workers)
    
    return results