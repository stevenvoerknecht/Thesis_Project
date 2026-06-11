import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Plot training metrics.")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()

def load_data(file_path: Path) -> pd.DataFrame:
    """Load CSV into Pandas DataFrame."""
    df = pd.read_csv(file_path)
    return df

def setup_style():
    """Set seaborn theme and matplotlib defaults."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 12})

def plot_metrics(df: pd.DataFrame, output_path: Optional[Path]):
    """
    Generate and save plots for Loss, Accuracy, and F1.
    Each metric will have train vs val in the same plot.
    """
    if df is None:
        return

    metrics = [
        ("Loss", "train_loss", "val_loss"),
        ("Accuracy", "train_acc", "val_acc"),
        ("F1-score", "train_f1", "val_f1")
    ]

    for title, train_col, val_col in metrics:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df[train_col], marker='o', label=f"Train {title}")
        plt.plot(df["epoch"], df[val_col], marker='o', label=f"Val {title}")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(f"{title} over Epochs")
        plt.legend()
        plt.grid(True)

        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path / f"{title.replace(' ', '_').lower()}.png")
        plt.show()
        plt.close()

def main():
    args = parse_args()
    setup_style()
    df = load_data(args.input_csv)
    plot_metrics(df, args.output_dir)

if __name__ == "__main__":
    main()
