from typing import List

import torch
import torch.nn as nn


class LateFusionMLP(nn.Module):
    def __init__(
        self,
        image_input_dim: int,
        text_input_dim: int,
        hidden_units: List[int],
        fusion_hidden: int,
        num_classes: int = 33,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        # Image branch
        self.image_net = self._make_branch(image_input_dim, hidden_units, dropout_rate)

        # Text branch
        self.text_net = self._make_branch(text_input_dim, hidden_units, dropout_rate)

        # Fusion network
        fusion_input_dim = hidden_units[-1] * 2
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden, num_classes),
        )

    def _make_branch(self, input_dim, hidden_units, dropout_rate):
        layers = []
        in_features = input_dim

        for hidden in hidden_units:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden

        return nn.Sequential(*layers)

    def forward(self, image_vec: torch.Tensor, text_vec: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_net(image_vec)
        text_feat = self.text_net(text_vec)

        fused = torch.cat([img_feat, text_feat], dim=1)
        return self.fusion_net(fused)
