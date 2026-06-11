from typing import Callable, Optional, Tuple, Dict, Any
import pickle
import pandas as pd
import torch
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class TCGADataset(Dataset):
    """
    Dataset class for TCGA multimodal embeddings (image + text) and cancer type labels.
    Ensures patient-aware splits to prevent data leakage.
    """
    def __init__(
        self,
        image_pickle_path: str,
        text_pickle_path: str,
        csv_path: str,
        config: Dict[str, Any],
        split_map_path: str = None,
        class_to_idx=None,
    ):
        """
        Args:
            image_pickle_path: Path to image embeddings pickle.
            text_pickle_path: Path to text embeddings pickle.
            csv_path: Path to full labels CSV.
            split_map_path: Optional path to CSV containing patient IDs for this split.
            class_to_idx: Optional dict mapping cancer types to integers.
        """
        # 1. Load embeddings
        with open(image_pickle_path, 'rb') as f:
            self.image_embeddings = pickle.load(f)
        with open(text_pickle_path, 'rb') as f:
            self.text_embeddings = pickle.load(f)
        
        # 2. Load labels and config
        self.labels_df = pd.read_csv(csv_path)
        self.config = config

        # 3. Create class mapping
        if class_to_idx is None:
            self.cancer_types = sorted(self.labels_df['cancer_type'].unique())
            self.class_to_idx = {name: i for i, name in enumerate(self.cancer_types)}
        else:
            self.class_to_idx = class_to_idx

        # 4. Filter patients with both embeddings and in CSV
        patients_with_images = set(self.image_embeddings.keys())
        patients_with_text = set(self.text_embeddings.keys())
        valid_patients = patients_with_images & patients_with_text & set(self.labels_df['patient_id'])

        if split_map_path and os.path.exists(split_map_path):
            split_df = pd.read_csv(split_map_path)
            split_patients = set(split_df['patient_id'])
            valid_patients = valid_patients & split_patients
            print(f"Loaded split from {split_map_path}: {len(valid_patients)} samples.")
        else:
            print(f"Warning: No split file provided. Loaded {len(valid_patients)} available samples.")

        self.valid_patient_ids = sorted(list(valid_patients))

        # 5. Map patient IDs to labels
        self.id_to_label = dict(zip(self.labels_df['patient_id'], self.labels_df['cancer_type']))

    def __len__(self):
        return len(self.valid_patient_ids)

    def __getitem__(self, idx):
        pid = self.valid_patient_ids[idx]

        # Get first embedding from image
        image_vector = self.image_embeddings[pid]['embeddings'][0]
        # Get first embedding from text
        text_vector = self.text_embeddings[pid]

        # if self.config["modality"] == "multimodal": 
        #     # Concatenate embeddings for multimodal input
        #     combined_vector = np.concatenate([image_vector, text_vector])

        #     # Convert to torch tensor
        #     x = torch.tensor(combined_vector, dtype=torch.float32)

        # elif self.config["modality"] == "text":
        #     x = torch.tensor(text_vector, dtype=torch.float32)

        # elif self.config["modality"] == "image":
        #     x = torch.tensor(image_vector, dtype=torch.float32)

        # Make the vectors a torch tensors
        image_vec = torch.tensor(image_vector, dtype=torch.float32)
        text_vec = torch.tensor(text_vector, dtype=torch.float32)

        # Get label
        label_name = self.id_to_label[pid]
        y = torch.tensor(self.class_to_idx[label_name], dtype=torch.long)

        return image_vec, text_vec, y