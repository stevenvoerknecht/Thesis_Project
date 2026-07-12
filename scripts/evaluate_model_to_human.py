import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score


def run_inference(model, tokenizer, texts, device, max_length=512, batch_size=16):
    """Batched model inference returning binary predictions (threshold = 0.5)."""
    model.eval()
    model.to(device)
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    probs_matrix = np.vstack(all_probs)
    preds_matrix = (probs_matrix >= 0.5).astype(int)
    return preds_matrix, probs_matrix


def compute_metrics(y_true, y_pred, label_names):
    """Computes Precision, Recall, F1, and Cohen's Kappa per class and macro averages."""
    per_class = []
    for i, name in enumerate(label_names):
        yt, yp = y_true[:, i], y_pred[:, i]
        prec = precision_score(yt, yp, pos_label=1, zero_division=0)
        rec = recall_score(yt, yp, pos_label=1, zero_division=0)
        f1 = f1_score(yt, yp, pos_label=1, zero_division=0)
        
        try:
            kappa = cohen_kappa_score(yt, yp)
        except Exception:
            kappa = np.nan

        per_class.append({
            "Narrative Dimension": name,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "Cohen's Kappa": round(kappa, 4) if not np.isnan(kappa) else 0.0,
            "Support": int(yt.sum())
        })

    df_per_class = pd.DataFrame(per_class)

    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    mean_kappa = np.nanmean([item["Cohen's Kappa"] for item in per_class])

    aggregates = {
        "Macro Precision": round(macro_prec, 4),
        "Macro Recall": round(macro_rec, 4),
        "Macro F1": round(macro_f1, 4),
        "Micro F1": round(micro_f1, 4),
        "Mean Cohen's Kappa": round(mean_kappa, 4)
    }

    return df_per_class, aggregates


def main():
    # Configuration
    model_name="Stevenvoerknecht/thesis-champion-model"
    human_csv="data/human/human_annotation.csv"
    llm_pqt="data/vllm_processed/labeled_subset_v1.pqt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    label_cols = [
        "elite_vs_mass_conflict",
        "in_group_vs_out_group_exclusion",
        "institutional_knowledge_denial",
        "societal_moral_regression",
        "imminent_acute_crisis_panic",
        "systemic_sovereignty_revival"
    ]

    # Load Parquet Dataset and track 0-based index
    print(f"Loading full LLM parquet dataset from: {llm_pqt}")
    df_llm = pd.read_parquet(llm_pqt)
    df_llm = df_llm.reset_index(drop=False).rename(columns={"index": "original_row_id"})

    # Ensure message_text exists (adjust column name if yours is 'text')
    text_col = "message_text" if "message_text" in df_llm.columns else "text"

    # Load Human Annotations
    print(f"Loading human annotations from: {human_csv}")
    df_human = pd.read_csv(human_csv)

    # Align Human CSV with Parquet using 'llm_row_nr' -> 'original_row_id'
    merged = pd.merge(
        df_human[["llm_row_nr"] + label_cols],
        df_llm[["original_row_id", text_col]],
        left_on="llm_row_nr",
        right_on="original_row_id",
        how="inner"
    )

    print(f"Successfully matched {len(merged)} human-annotated messages.")

    if len(merged) == 0:
        raise ValueError("No matching rows found between human 'llm_row_nr' and parquet index!")

    # Extract Text & Ground Truth (Binarize Likert ratings)
    texts = merged[text_col].tolist()
    
    y_true_human = np.zeros((len(merged), len(label_cols)), dtype=int)
    for i, col in enumerate(label_cols):
        y_true_human[:, i] = (pd.to_numeric(merged[col], errors='coerce').fillna(0) >= 2).astype(int)

    # Load Transformer Model & Tokenizer
    print(f"\nLoading model and tokenizer from '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Run Live Inference on the Human Text Sample
    print(f"Running inference on {len(texts)} human-annotated texts...")
    y_pred_model, _ = run_inference(model, tokenizer, texts, device)

    # Compute & Display Performance Metrics
    df_class_human, agg_human = compute_metrics(y_true_human, y_pred_model, label_cols)

    print("\n" + "="*20 + f" TRANSFORMER MODEL vs. HUMAN LABELS (N={len(merged)}) " + "="*20)
    print(df_class_human.to_string(index=False))
    print("\nAggregate Metrics:", agg_human)
    
    # Export LaTeX Table directly
    latex_filename = "table_model_vs_human_direct.tex"
    df_class_human.to_latex(latex_filename, index=False)
    print(f"\nSaved LaTeX table to '{latex_filename}'")


if __name__ == "__main__":
    main()