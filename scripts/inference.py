import argparse
from pathlib import Path
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from tqdm import tqdm


class TextDataset(Dataset):
    """A lightweight Dataset designed to hold raw streaming text for inference."""
    def __init__(self, texts, tokenizer, max_length: int = 256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Padding handled dynamically by the collator
            return_attention_mask=True,
        )
        item = {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long)
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = torch.tensor(encoding["token_type_ids"], dtype=torch.long)
        return item


def run_inference(model_dir: str, data_path: str, output_path: str, batch_size: int = 32, max_length: int = 256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the fine-tuned model and matching tokenizer
    print(f"Loading best checkpoint weights from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Define your explicit target structure matching your training setup
    label_columns = [
        'elite_vs_mass_conflict',
        'in_group_vs_out_group_exclusion',
        'institutional_knowledge_denial',
        'societal_moral_regression',
        'imminent_acute_crisis_panic',
        'systemic_sovereignty_revival'
    ]

    # 2. Read the unseen data efficiently using Polars
    print(f"Streaming unseen dataset from: {data_path}")
    df_lazy = pl.scan_parquet(data_path)
    
    # We collect only the unique identifier and text to optimize memory
    # Missing texts are filled with empty strings to prevent pipeline breakages
    df = df_lazy.select(["message_text"]).collect()
    texts = df["message_text"].fill_null("").cast(pl.String).to_list()
    
    # 3. Create Dataset and DataLoader instances
    dataset = TextDataset(texts, tokenizer, max_length=max_length)
    
    # Use the native transformers collator implementation dynamically
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=data_collator,
        num_workers=2 if device.type == "cuda" else 0
    )

    # 4. Batched Inference Loop
    all_predictions = []
    print(f"Starting prediction loop over {len(texts)} entries...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Move batch items to target architecture device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

            # Forward pass through your classification heads
            outputs = model(**kwargs)
            logits = outputs.logits
            
            # Map raw model outputs through a Sigmoid activation matrix
            probs = torch.sigmoid(logits)
            
            # Binary mask assignments thresholded at alpha >= 0.5
            preds = (probs >= 0.5).int().cpu().numpy()
            all_predictions.append(preds)

    # Concatenate the array blocks into a single prediction matrix
    final_preds = np.vstack(all_predictions) if len(all_predictions) > 0 else np.empty((0, len(label_columns)))

    # 5. Re-integrate predictions and write back to disk
    print("Structuring final data matrix...")
    pred_dict = {label_columns[i]: final_preds[:, i] for i in range(len(label_columns))}
    
    # Re-read the full original source file and attach predictions
    original_df = pl.read_parquet(data_path)
    output_df = original_df.with_columns([
        pl.series(name, pred_dict[name]) for name in label_columns
    ])

    print(f"Writing outputs to high-performance Parquet: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.write_parquet(output_path)
    print("Inference execution finished successfully!")


if __name__ == "__main__":
    import numpy as np
    parser = argparse.ArgumentParser(description="Run batched multi-label narrative inference.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to your saved best_model directory")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the unseen .pqt file")
    parser.add_argument("--output_path", type=str, default="data/predictions/output.parquet", help="Where to save the tagged results")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for execution acceleration")
    parser.add_argument("--max_length", type=int, default=256, help="Tokenizer sequence length limit")
    args = parser.parse_args()

    run_inference(args.model_dir, args.data_path, args.output_path, args.batch_size, args.max_length)