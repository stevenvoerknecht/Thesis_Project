import argparse
import itertools
import copy
from pathlib import Path
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import f1_score
from ml_core.utils import load_config, seed_everything, setup_logger
from ml_core.data import get_dataloaders 

logger = setup_logger("Narrative_Trainer")

def compute_metrics(eval_pred):
    """Computes Macro and Micro F1 scores for Multi-Label tracking."""
    predictions, labels = eval_pred
    # Apply sigmoid to convert raw logits to probabilities
    probs = 1 / (1 + np.exp(-predictions))
    # Threshold at 0.5 to binarize predictions
    preds = (probs > 0.5).astype(int)
    
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "micro_f1": f1_score(labels, preds, average="micro")
    }

def main(args):
    base_config = load_config(args.config)
    seed_everything(base_config["seed"])
    
    # 1. Load Tokenizer & Datasets
    model_name = base_config["model_name"] 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Custom function that loads your JSON files and returns Hugging Face Dataset objects
    train_dataset, val_dataset = get_dataloaders(base_config, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 2. Extract Hyperparameter Grid
    lr_list = base_config["training"]["learning_rates"]
    batch_list = base_config["training"]["batch_sizes"]
    grid = list(itertools.product(lr_list, batch_list))

    best_macro_f1 = -1.0

    # 3. HPO Optimization Loop
    for run_idx, (lr, batch_size) in enumerate(grid, 1):
        print(f"\n=== Run {run_idx}/{len(grid)} | LR: {lr} | Batch Size: {batch_size} ===")

        # Instantiate a clean, multi-label text classification model
        # num_labels = 6 parents + your child categories
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=base_config["num_labels"],
            problem_type="multi_label_classification" # Automatically invokes BCEWithLogitsLoss
        )

        run_dir = Path(base_config["training"]["save_dir"]) / f"run_{run_idx}"

        # Define specialized Transformer arguments
        training_args = TrainingArguments(
            output_dir=str(run_dir),
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=base_config["training"]["epochs"],
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            fp16=torch.cuda.is_available(), # Mixed precision training if GPU is available
            report_to="none"
        )

        # Utilize native HF Trainer instead of custom ml_core.solver
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        
        # Evaluate performance
        eval_results = trainer.evaluate()
        current_f1 = eval_results["eval_macro_f1"]
        print(f"Run {run_idx} Finished. Macro F1: {current_f1:.4f}")

        if current_f1 > best_macro_f1:
            best_macro_f1 = current_f1
            print(f" New best configuration found!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hierarchical Multi-Label Transformer")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    main(args)