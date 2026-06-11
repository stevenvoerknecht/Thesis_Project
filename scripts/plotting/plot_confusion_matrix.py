import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    train_loader, val_loader = get_dataloaders(config)

    # Build model
    model_cfg = config["model"]
    model = MLP(**model_cfg).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix - Champion Model")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output)
    plt.show()

    print(f"Confusion matrix saved to: {args.output}")


if __name__ == "__main__":
    main()
