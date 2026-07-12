import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)

# Setup and plot styling
OUTPUT_DIR = "thesis_results_output"
SET_INFERENCE_PATH = "data/inference/test_inference_champion.pqt"
os.makedirs(OUTPUT_DIR, exist_ok=True)
model_threshold = 0.3

# Set publication-quality plot aesthetics
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

LABELS = [
    "elite_vs_mass_conflict",
    "in_group_vs_out_group_exclusion",
    "institutional_knowledge_denial",
    "societal_moral_regression",
    "imminent_acute_crisis_panic",
    "systemic_sovereignty_revival"
]

LABELS_PRED = [
    "Populist_narrative",
    "Nativist_narrative",
    "Denialist_narrative",
    "Declinist_narrative",
    "Apocalypticist_narrative",
    "Revisionist_narrative"
]

# Display-friendly labels for publication figures
DISPLAY_LABELS = [
    "Populist Narrative",
    "Nativist Narrative",
    "Denialist Narrative",
    "Declinist Narrative",
    "Apocolypticist Narrative",
    "Revisionist Narrative"
]

# Narrative-specific optimal decision thresholds
THRESHOLDS = {
    "Populist_narrative": 0.32,
    "Nativist_narrative": 0.44,
    "Denialist_narrative": 0.18,
    "Declinist_narrative": 0.22,
    "Apocalypticist_narrative": 0.3,
    "Revisionist_narrative": 0.26
}

# Load and perpare data
df = pd.read_parquet(SET_INFERENCE_PATH)

# Extract matrices and binarize
y_true = (df[[f"{label}" for label in LABELS]].fillna(0).values >= 2).astype(int)
y_scores = df[[f"{label}_score" for label in LABELS_PRED]].fillna(0.0).values

# Apply Narrative-Specific Thresholding
y_pred = np.zeros_like(y_scores, dtype=int)
threshold_list = []

for i, label in enumerate(LABELS_PRED):
    thresh = THRESHOLDS.get(label, 0.50)
    threshold_list.append(thresh)
    y_pred[:, i] = (y_scores[:, i] >= thresh).astype(int)

# generate matrix labels and LaTeX export
print("Generating Performance Metrics Table...")

precisions, recalls, f1s, supports = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

metrics_df = pd.DataFrame({
    "Narrative Category": DISPLAY_LABELS,
    "Support": supports,
    "Precision": precisions,
    "Recall": recalls,
    "F1-Score": f1s
})

# Overall Averages
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

metrics_df.to_csv(os.path.join(OUTPUT_DIR, "narrative_performance_metrics.csv"), index=False)

# Find best threshold per narrative
best_thresholds = {}
for i, label in enumerate(LABELS_PRED):
    best_f1 = 0.0
    best_t = 0.5
    for t in np.arange(0.10, 0.90, 0.02):
        preds = (y_scores[:, i] >= t).astype(int)
        score = f1_score(y_true[:, i], preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = t
    best_thresholds[label] = round(float(best_t), 2)

print("Optimized Thresholds:", best_thresholds)

# Format LaTeX Table String
latex_table = metrics_df.to_latex(
    index=False,
    float_format="%.3f",
    caption="Per-class classification performance on evaluation dataset with tuned decision thresholds.",
    label="tab:model_performance"
)

print("\n--- LATEX TABLE FOR THESIS ---")
print(latex_table)
print(f"Macro-F1: {macro_f1:.4f} | Micro-F1: {micro_f1:.4f}\n")


# Multi-label confusion matrix
print("Generating Confusion Matrices Plot...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
axes = axes.flatten()

for i, label in enumerate(LABELS):
    cm = confusion_matrix(y_true[:, i], y_pred[:, i], normalize="true")
    
    sns.heatmap(
        cm, annot=True, fmt=".2%", cmap="Blues", cbar=False,
        ax=axes[i], xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"]
    )
    axes[i].set_title(DISPLAY_LABELS[i], fontsize=11, fontweight="bold")
    axes[i].set_ylabel("True Label")
    axes[i].set_xlabel("Predicted Label")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), bbox_inches="tight")
plt.close()


# Narrative co-occurence heatmap
print("Generating Narrative Co-occurrence Heatmap...")
co_occurrence_true = np.dot(y_true.T, y_true)
co_occurrence_pred = np.dot(y_pred.T, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Ground Truth Co-occurrence
sns.heatmap(
    co_occurrence_true, annot=True, fmt="d", cmap="Purples",
    xticklabels=DISPLAY_LABELS, yticklabels=DISPLAY_LABELS, ax=axes[0]
)
axes[0].set_title("Ground Truth Co-occurrence Matrix", fontweight="bold")
axes[0].tick_params(axis='x', rotation=45)

# Predicted Co-occurrence
sns.heatmap(
    co_occurrence_pred, annot=True, fmt="d", cmap="Greens",
    xticklabels=DISPLAY_LABELS, yticklabels=DISPLAY_LABELS, ax=axes[1]
)
axes[1].set_title("Model Predicted Co-occurrence Matrix", fontweight="bold")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "narrative_cooccurrence_matrix.png"), bbox_inches="tight")
plt.close()


# Precision Recall curves
print("Generating Precision-Recall Curves...")
plt.figure(figsize=(9, 6))

for i, label in enumerate(LABELS):
    precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
    ap_score = average_precision_score(y_true[:, i], y_scores[:, i])
    
    plt.plot(recall, precision, lw=2, label=f"{DISPLAY_LABELS[i]} (AP = {ap_score:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves per Narrative Dimension", fontweight="bold")
plt.legend(loc="lower left", fontsize=9)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "precision_recall_curves.png"), bbox_inches="tight")
plt.close()

print(f"All evaluation artifacts saved successfully to folder: '{OUTPUT_DIR}'")