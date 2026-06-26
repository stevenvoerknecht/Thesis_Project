import argparse
import itertools
import copy
import os
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
from ml_core.utils import load_config, seed_everything
from ml_core.data import NarrativeParquetDataset

def compute_metrics(eval_pred):
    """Computes Macro and Micro F1 scores for Multi-Label tracking."""
    predictions, labels = eval_pred
    # Apply sigmoid to convert raw logits to probabilities
    probs = 1 / (1 + np.exp(-predictions))
    # Threshold at 0.5 to binarize predictions
    preds = (probs > 0.5).astype(int)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1
    }

def main(args):
    base_config = load_config(args.config)
    seed_everything(base_config["seed"])
    
    # Load Tokenizer 
    model_name = base_config["model_name"] 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the datasets
    input_path = Path(base_config["data"]["processed_dir"])
    train_path = input_path / 'train_split.pqt'
    val_path = input_path / 'val_split.pqt'
    train_dataset = NarrativeParquetDataset(train_path, tokenizer, max_length=512)
    val_dataset = NarrativeParquetDataset(val_path, tokenizer, max_length=512)
    
    # id2label for correct label names
    id2label = {
        0: "Populist_narrative",
        1: "Nativist_narrative",
        2: "Denialist_narrative",
        3: "Declinist_narrative",
        4: "Apocalyptist_narrative",
        5: "Revisionist_narrative"
    }
    label2id = {v: k for k, v in id2label.items()}

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create the correct MLFlow tracking URI
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"

    # Extract Hyperparameter Grid
    lr_list = base_config["training"]["learning_rate"]
    batch_list = base_config["training"]["batch_size"]
    wd_list = base_config["training"]["weight_decay"]

    # If they are single scalars (int/float), wrap them in a list so itertools works
    if not isinstance(lr_list, list):
        lr_list = [lr_list]
    if not isinstance(batch_list, list):
        batch_list = [batch_list]
    if not isinstance(wd_list, list):
        wd_list = [wd_list]

    grid = list(itertools.product(lr_list, batch_list, wd_list))

    best_macro_f1 = -1.0

    # HPO Optimization Loop
    for run_idx, (lr, batch_size, wd) in enumerate(grid, 1):
        print(f"\n=== Run {run_idx}/{len(grid)} | Learning Rate: {lr} | Batch Size: {batch_size} | Weight Decay: {wd} ===")

        # Instantiate a clean, multi-label text classification model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=base_config["num_labels"],
            problem_type="multi_label_classification", # Automatically invokes BCEWithLogitsLoss
            id2label=id2label,
            label2id=label2id
        )

        run_dir = Path(base_config["training"]["save_dir"]) / f"run_{run_idx}"

        # Define specialized Transformer arguments
        training_args = TrainingArguments(
            output_dir=str(run_dir),
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=base_config["training"]["epochs"],
            weight_decay=wd,
            eval_strategy="epoch",
            save_strategy="epoch",
            fp16=torch.cuda.is_available(), # Mixed precision training if GPU is available
            report_to=["mlflow", "tensorboard"],
            run_name=f"run_lr_{lr}_bs_{batch_size}",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=1
        )

        # Utilize native HF Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # Extract the best model from RAM and save it
        best_model_path = Path('experiments/result_champion')
        trainer.save_model(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        
        # Evaluate performance
        eval_results = trainer.evaluate()
        current_f1 = eval_results["eval_macro_f1"]
        print(f"Run {run_idx} Finished. Macro F1: {current_f1:.4f}. Best weights saved to: {best_model_path}")

        if current_f1 > best_macro_f1:
            best_macro_f1 = current_f1
            print(f"New best configuration found!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hierarchical Multi-Label Transformer")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    main(args)