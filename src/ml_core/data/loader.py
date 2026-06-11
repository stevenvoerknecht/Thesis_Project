from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import os
import torch
import random
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from collections import Counter

from .tcga import TCGADataset

# Function to easily get loaders for the rest of the team
def get_dataloaders(config):
    data_cfg = config["data"]
    seed = config.get("seed", 42)
    base_path = Path(data_cfg["data_path"])

    def seed_worker(worker_id):
        # seed every worker for reproducibility
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Hardcoded paths relative to the project root
    raw_dir = 'data/raw'
    processed_dir = 'data/processed'
    
    image_pkl_path = os.path.join(raw_dir, 'tcga_titan_embeddings.pkl')
    text_pkl_path = os.path.join(raw_dir, 'text_embeddings.pkl')
    labels_path = os.path.join(raw_dir, 'tcga_patient_to_cancer_type.csv')
    
    train_split = os.path.join(processed_dir, 'train_split.csv')
    val_split = os.path.join(processed_dir, 'val_split.csv')

    labels_df = pd.read_csv(labels_path)
    cancer_types = sorted(labels_df['cancer_type'].unique())
    class_to_idx = {name: i for i, name in enumerate(cancer_types)}
    print("Number of classes:", len(class_to_idx))
    
    # 1. Create Train Dataset
    print("Initializing Train Dataset...")
    train_dataset = TCGADataset(image_pkl_path, text_pkl_path, labels_path, config, split_map_path=train_split, 
                                class_to_idx=class_to_idx)
    
    # 2. Create Validation Dataset (reuse class_to_idx from train to ensure mapping matches!)
    print("Initializing Val Dataset...")
    val_dataset = TCGADataset(image_pkl_path, text_pkl_path, labels_path, config, split_map_path=val_split, 
                              class_to_idx=class_to_idx)
    
    # 3. Create a sampler for the train dataset
    labels = np.array([train_dataset.class_to_idx[train_dataset.id_to_label[pid]]for pid in train_dataset.valid_patient_ids])
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_dataset),
        replacement=True
    )
    # 4. Create a generator for the dataloader
    g = torch.Generator()
    g.manual_seed(seed)

    # 5. Create Loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=data_cfg["batch_size"], 
        sampler=sampler, 
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=False, 
        num_workers=data_cfg.get("num_workers", 0)
        )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=data_cfg["batch_size"], 
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=False, 
        num_workers=data_cfg.get("num_workers", 0)
        )
    
    return train_loader, val_loader
