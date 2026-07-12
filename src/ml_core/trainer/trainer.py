import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

# Define Multi-Label Binary Focal Loss
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_weight = alpha_factor * focal_weight

        loss = focal_weight * bce_loss
        return loss.mean()


# Subclass HuggingFace Trainer to override compute_loss
class FocalLossTrainer(Trainer):
    def __init__(self, gamma=2.0, alpha=0.25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fct = BinaryFocalLoss(gamma=gamma, alpha=alpha)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.focal_loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss