from typing import Callable, Optional, Tuple, Dict, Any
import pickle
import pandas as pd
import torch
import os
import numpy as np
import torch
from torch.utils.data import Dataset

# A lightweight multi-label Dataset wrapper for Hugging Face tokenizers
class NarrativeParquetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):

        self.texts = df['message_text'].fillna("").astype(str).tolist()
        
        # Define explicit target columns based the LLM-generated schema
        self.label_columns = [
            'elite_vs_mass_conflict',
            'in_group_vs_out_group_exclusion',
            'institutional_knowledge_denial',
            'societal_moral_regression',
            'imminent_acute_crisis_panic',
            'systemic_sovereignty_revival'
            # TODO: Append child narrative columns here once generated, e.g.:
            # '1.1_institutional_betrayal', '1.2_media_capture', etc.
        ]
    
        # Convert scores >= 2 (Moderate/Severe) to an active binary state (1.0).
        raw_labels = df[self.label_columns].values
        self.labels = (raw_labels >= 2).astype(np.float32)
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Flatten out the batch dimension added by return_tensors
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(labels, dtype=torch.float32)
        
        return item