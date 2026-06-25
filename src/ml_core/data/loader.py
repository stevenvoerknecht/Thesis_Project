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

from .dataset import NarrativeParquetDataset

# Function to easily get loaders for the rest of the team
def get_dataloaders(config):
    data_cfg = config["data"]
    seed = config.get("seed", 42)
    
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    model_name = config.get("model_name", "microsoft/deberta-v3-base")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Resolve paths relative to your config
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    train_parquet_path = processed_dir / 'train_split.parquet'
    val_parquet_path = processed_dir / 'val_split.parquet'

    print(f"Loading Telegram Parquet datasets from {processed_dir}...")
    train_df = pd.read_parquet(train_parquet_path)
    val_df = pd.read_parquet(val_parquet_path)

    # Instantiate datasets
    max_len = data_cfg.get("max_length", 256)
    train_dataset = NarrativeParquetDataset(train_df, tokenizer, max_length=max_len)
    val_dataset = NarrativeParquetDataset(val_df, tokenizer, max_length=max_len)
    
    # Dynamic label count validation for your train.py configuration
    config["num_labels"] = len(train_dataset.label_columns)
    print(f"Configured for multi-label tracking across {config['num_labels']} distinct targets.")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=data_cfg["batch_size"], 
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True, 
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