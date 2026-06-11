import argparse
import itertools
import copy

import torch
import torch.optim as optim
from pathlib import Path
from ml_core.data import get_dataloaders
from ml_core.models import LateFusionMLP
from ml_core.solver import Trainer
from ml_core.utils import load_checkpoint, load_config, seed_everything, setup_logger

logger = setup_logger("Experiment_Runner")


def main(args):
     # 1. Load Config & Set Seed
    base_config = load_config(args.config)
    name = base_config["experiment_name"]
    seed = base_config["seed"]

    seed_everything(seed)
    print(f"Experiment {name} has started")
    print("Configuration has loaded and seeds have been set")

    # 2. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3. Data (only load once)
    train_loader, val_loader = get_dataloaders(base_config)
    print("Dataloaders are ready")
    print(f"Train samples: {len(train_loader.dataset)}", flush=True)
    print(f"Val samples:   {len(val_loader.dataset)}", flush=True)

    # Get input dimension
    # x, y = next(iter(train_loader))
    # input_dim = x.shape[1]

    # 4. Build hyperparameter grid
    model_cfg = base_config["model"]
    training_cfg = base_config["training"]

    hidden_units_list = model_cfg["hidden_units"]
    dropout_list = model_cfg["dropout_rate"]
    lr_list = training_cfg["learning_rate"]
    optim_list = base_config["optimizer"]

    # Ensure everything is list
    if not isinstance(hidden_units_list, list) or not isinstance(hidden_units_list[0], list):
        hidden_units_list = [hidden_units_list]
    if not isinstance(dropout_list, list):
        dropout_list = [dropout_list]
    if not isinstance(lr_list, list):
        lr_list = [lr_list]
    if not isinstance(optim_list, list):
        optim_list = [optim_list]

    grid = list(itertools.product(hidden_units_list, dropout_list, lr_list, optim_list))

    print(f"Running {len(grid)} hyperparameter configurations...")

    best_val_loss = float("inf")
    best_checkpoint_path = None

    # 5. Loop over configs
    for run_idx, (hidden_units, dropout, lr, opt_name) in enumerate(grid, 1):
        print(f"\n=== Run {run_idx}/{len(grid)} ===")
        print(f"hidden_units={hidden_units}, dropout={dropout}, lr={lr}, optimizer={opt_name}")

        # Create run-specific config
        config = copy.deepcopy(base_config)
        # config["model"]["input_dim"] = input_dim
        config["model"]["hidden_units"] = hidden_units
        config["model"]["dropout_rate"] = dropout
        config["training"]["learning_rate"] = lr
        config["optimizer"] = opt_name

        # 6. Model
        model = LateFusionMLP(**config["model"]).to(device)

        # 7. Optimizer
        opt_name_l = opt_name.lower()
        if opt_name_l == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif opt_name_l == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        elif opt_name_l == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        # 8. Trainer & Fit
        trainer = Trainer(model, optimizer, config, device)
        trainer.fit(train_loader, val_loader, start_epoch=0)

        # Check if this run is best
        if trainer.best_val_loss < best_val_loss:
            best_val_loss = trainer.best_val_loss
            best_checkpoint_path = (
                Path(config["training"]["save_dir"]) / "best_checkpoint.pt"
            )

    print("\n=== HPO finished ===")
    print(f"Best val loss: {best_val_loss}")
    if best_checkpoint_path:
        print(f"Best checkpoint at: {best_checkpoint_path}")


if __name__ == "__main__":
    print("Run has started")
    parser = argparse.ArgumentParser(description="Train a multimodal MLP on TCGA")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    main(args)
    print("Full run completed")
