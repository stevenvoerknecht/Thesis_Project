import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score


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
    test_set_path = "data/inference/test_inference_focalloss.pqt"
    llm_set_path = "data/vllm_processed/labeled_subset_v1.pqt"

    # Core narrative label categories
    label_cols_pred = [
        "Populist_narrative",
        "Nativist_narrative",
        "Denialist_narrative",
        "Declinist_narrative",
        "Apocalypticist_narrative",
        "Revisionist_narrative"
    ]

    label_cols_llm = [
        "elite_vs_mass_conflict",
        "in_group_vs_out_group_exclusion",
        "institutional_knowledge_denial",
        "societal_moral_regression",
        "imminent_acute_crisis_panic",
        "systemic_sovereignty_revival"
    ]

    # Model boolean prediction columns according to your schema
    model_pred_cols = [f"{col}_active" for col in label_cols_pred]

    # Load Parquet Test Set
    print(f"Loading test set: {test_set_path}")
    df_test = pd.read_parquet(test_set_path)

    # Load Parquet LLM-labelled set
    print(f"Loading LLM-labelled set: {llm_set_path}")
    df_llm = pd.read_parquet(llm_set_path)

    # Convert model active booleans to binary ints (0 or 1)
    y_pred_model = df_test[model_pred_cols].astype(int).values

    # Evaluation: Model Predictions vs. LLM Silver Labels
    # Convert LLM labels (Int64/Likert) to binary target (>0)
    y_true_llm = (df_test[label_cols_llm].fillna(0).values > 0).astype(int)
    
    df_class_llm, agg_llm = compute_metrics(y_true_llm, y_pred_model, label_cols_llm)

    print("\n" + "="*20 + " MODEL vs. LLM SILVER LABELS " + "="*20)
    print(df_class_llm.to_string(index=False))
    print("\nAggregate Metrics (Model vs. LLM):", agg_llm)

if __name__ == "__main__":
    main()