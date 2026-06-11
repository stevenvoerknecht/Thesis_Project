import csv
from pathlib import Path
from typing import Any, Dict
import yaml

# Optional: TensorBoard logging
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        base_dir: str = "experiments/results",
    ):
        self.run_dir = Path(base_dir) / experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save config to YAML
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

        # Initialize CSV
        self.csv_path = self.run_dir / "metrics.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "epoch", 
            "train_loss", "train_acc", "train_f1",
            "val_loss", "val_acc", "val_f1", "val_f2",
            "val_roc_auc", "val_pr_auc"
        ])

        # TensorBoard
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.run_dir))
        else:
            self.writer = None

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Log metrics to CSV and TensorBoard
        """
        row = [epoch] + [metrics.get(k, float('nan')) for k in [
            "train_loss","train_acc","train_f1",
            "val_loss","val_acc","val_f1","val_f2",
            "val_roc_auc","val_pr_auc"
        ]]
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, epoch)

    def get_checkpoint_path(self, filename: str) -> str:
        return str(self.run_dir / filename)

    def close(self):
        self.csv_file.close()
        if self.writer is not None:
            self.writer.close()
