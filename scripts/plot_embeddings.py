import argparse
import torch
import matplotlib.pyplot as plt
import umap
from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.utils import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="scripts/plots/champion/umap.png")
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data (val set is enough)
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

    features = []
    labels_all = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Use logits as features (or adapt if you expose hidden layer)
            feats = outputs.cpu()

            features.append(feats)
            labels_all.append(labels)

    features = torch.cat(features).numpy()
    labels_all = torch.cat(labels_all).numpy()

    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    emb_2d = reducer.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels_all, cmap="tab20", s=10)
    plt.colorbar(scatter, label="Class")
    plt.title("UMAP of Champion Model Embeddings")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    plt.show()

    print(f"Saved UMAP plot to {args.output}")

if __name__ == "__main__":
    main()
