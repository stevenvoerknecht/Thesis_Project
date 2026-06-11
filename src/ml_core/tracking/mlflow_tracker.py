import mlflow
import mlflow.pytorch
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any


class MLflowTracker:
    def __init__(self, config: Dict[str, Any]):

        self.config = config

        # read the config for the name and directory
        experiment_name = config['mlflow']['experiment_name']
        tracking_dir = config['mlflow']['tracking_dir']
        tracking_dir = os.path.abspath(tracking_dir)
        os.makedirs(tracking_dir, exist_ok=True)

        # lokaal, file-based (Snellius-proof)
        mlflow.set_tracking_uri(f"file://{tracking_dir}")
        mlflow.set_experiment(experiment_name)

        # Stop actieve run als er al een is
        if mlflow.active_run() is not None:
            mlflow.end_run()

        self.run = mlflow.start_run()

        # log volledige config
        mlflow.log_params(self._flatten_dict(config))

        # log git commit
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()
        mlflow.log_param("git_commit", git_commit)

        # log environment
        reqs = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"]
        ).decode("utf-8")

        Path("requirements_logged.txt").write_text(reqs)
        mlflow.log_artifact("requirements_logged.txt")

    def log_metrics(self, metrics: Dict[str, float], step: int):
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_model(self, model, artifact_path: str):
        mlflow.pytorch.log_model(model, artifact_path)
    
    def log_artifact(self, file_path: str, artifact_path: str = None):
        mlflow.log_artifact(file_path, artifact_path=artifact_path)

    def end(self):
        mlflow.end_run()

    def _flatten_dict(self, d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
