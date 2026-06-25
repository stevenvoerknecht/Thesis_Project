from typing import Optional
from pathlib import Path
import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset

class NarrativeParquetDataset(Dataset):
    def __init__(self, parquet_path: Path, tokenizer, max_length: int = 512):
        """
        A memory-efficient Dataset wrapper that loads data directly from a Parquet file 
        using Polars and handles multi-label binarization.
        """
        # Read Parquet using Polars
        df = pl.read_parquet(parquet_path)
        
        # Extract and clean text column
        texts = df["message_text"].fill_null("").cast(pl.String).to_list()
        
        # Define target columns 
        self.label_columns = [
            'elite_vs_mass_conflict',
            'in_group_vs_out_group_exclusion',
            'institutional_knowledge_denial',
            'societal_moral_regression',
            'imminent_acute_crisis_panic',
            'systemic_sovereignty_revival'
            # Append child sub-narratives here as the taxonomy gets expended
        ]
    
        # Convert ordinal scores >= 2 to an active binary float state matrix
        raw_labels = df[self.label_columns].to_numpy()
        self.labels = (raw_labels >= 2).astype(np.float32)
        
        # Tokenize the entire corpus
        print(f"Tokenizing {len(texts)} samples from {Path(parquet_path).name}...")
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        # Pull pre-tokenized sequences directly from lists
        item = {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long)
        }
        
        # If tokenizer variant outputs token_type_ids (DeBERTa), forward them too
        if "token_type_ids" in self.encodings:
            item["token_type_ids"] = torch.tensor(self.encodings["token_type_ids"][idx], dtype=torch.long)
            
        # Add targets matrix slice as a tensor
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return item