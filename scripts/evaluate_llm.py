import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score

def evaluate_llm_annotations():
    """
    Evaluates LLM multi-label predictions against a human-annotated ground truth sample.
    """
    # Load Datasets
    llm_file_path = "data/vllm_processed/labeled_subset_v1.pqt"
    human_annotation_path = "data/human/human_annotation.csv"
    df_llm = pd.read_parquet(llm_file_path)
    df_human = pd.read_csv(human_annotation_path)

    row_id_col = "llm_row_nr"

    # List of your target narrative dimensions
    label_cols = [
        "elite_vs_mass_conflict",
        "in_group_vs_out_group_exclusion",
        "institutional_knowledge_denial",
        "societal_moral_regression",
        "imminent_acute_crisis_panic",
        "systemic_sovereignty_revival"
    ]
    # Add an explicit original_row_id column to the LLM dataset matching its 0-indexed position
    df_llm = df_llm.reset_index(drop=False).rename(columns={"index": "original_row_id"})
    
    # Ensure integer data types for merging
    df_llm["original_row_id"] = df_llm["original_row_id"].astype(int)
    df_human[row_id_col] = df_human[row_id_col].astype(int)

    # Merge on Row Index
    merged_df = pd.merge(
        df_human[[row_id_col] + label_cols],
        df_llm[["original_row_id"] + label_cols],
        left_on=row_id_col,
        right_on="original_row_id",
        suffixes=("_human", "_llm")
    )

    print(f"Successfully aligned {len(merged_df)} instances based on row index.\n")

    # Extract numpy matrices and binarize for metric calculation (Likert scale >=2)
    y_true = (merged_df[[f"{col}_human" for col in label_cols]].values >= 2).astype(int)
    y_pred = (merged_df[[f"{col}_llm" for col in label_cols]].values >= 2).astype(int)

    # Calculate Per-Class Metrics
    per_class_results = []
    
    for i, col in enumerate(label_cols):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        prec = precision_score(y_t, y_p, pos_label=1, zero_division=0)
        rec = recall_score(y_t, y_p, pos_label=1, zero_division=0)
        f1 = f1_score(y_t, y_p, pos_label=1, zero_division=0)
        kappa = cohen_kappa_score(y_t, y_p)

        per_class_results.append({
            "Narrative Dimension": col,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "Cohen's Kappa": round(kappa, 4),
            "Human Annotated": int(y_t.sum())
        })

    df_per_class = pd.DataFrame(per_class_results)

    # Calculate Aggregate Metrics
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Mean Kappa across categories
    mean_kappa = np.mean([res["Cohen's Kappa"] for res in per_class_results])

    # Print Summary Tables for Thesis Presentation
    print("=================== PER-CLASS EVALUATION ===================")
    print(df_per_class.to_string(index=False))
    
    print("\n=================== OVERALL AGGREGATE METRICS ===================")
    print(f"Macro Precision : {macro_prec:.4f}")
    print(f"Macro Recall    : {macro_rec:.4f}")
    print(f"Macro F1-Score  : {macro_f1:.4f}")
    print(f"Micro F1-Score  : {micro_f1:.4f}")
    print(f"Mean Cohen's K  : {mean_kappa:.4f}")

    # Convert per-class results to LaTeX Table
    latex_table = df_per_class.to_latex(
        index=False,
        caption="Evaluation of LLM Zero-Shot Annotations against Sampled Human Benchmark ($N=100$)",
        label="tab:llm_human_eval",
        column_format="lccccc"
    )

    with open("llm_eval_results.tex", "w") as f:
        f.write(latex_table)

    print("\nSaved LaTeX table to 'llm_eval_results.tex'")

    return df_per_class, {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "mean_kappa": mean_kappa
    }

if __name__ == "__main__":
    evaluate_llm_annotations()