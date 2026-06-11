import argparse
import torch
from sklearn.metrics import f1_score, classification_report
from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.utils import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data (use val loader for evaluation)
    _, val_loader = get_dataloaders(config)

    # Get input dim
    x, _ = next(iter(val_loader))
    input_dim = x.shape[1]
    config["model"]["input_dim"] = input_dim

    # Build model
    model = MLP(**config["model"]).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu()

            all_preds.append(preds)
            all_targets.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    micro_f1 = f1_score(all_targets, all_preds, average="micro")

    print("=== F1 Scores ===")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Micro-F1: {micro_f1:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(all_targets, all_preds))

if __name__ == "__main__":
    main()
