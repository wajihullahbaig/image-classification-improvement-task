"""Multi-Layer Perceptron (MLP) classifier for flattened raw or GIST vectors."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Three-layer feed-forward network with ReLU activations and dropout."""

    def __init__(
        self,
        input_length: int,
        hidden_dim: int = 512,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.input_length)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)
