import logging
import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml


def setup_logger(name: str = "MLOps_Course") -> logging.Logger:
    """Configures a standardized logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_config(path: str) -> Dict[str, Any]:
    """Safely loads a yaml configuration file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)

        def expand(value):
            if isinstance(value, str):
                return os.path.expandvars(value)
            return value

        for section in config.values():
            if isinstance(section, dict):
                for k, v in section.items():
                    section[k] = expand(v)

    return config


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint["epoch"] + 1
    return start_epoch, checkpoint


def seed_everything(seed: int):
    """Ensures reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Using deterministic execution
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
