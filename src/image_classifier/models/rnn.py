"""Recurrent Neural Network (RNN) classifier for sequential image rows or features."""

import torch
import torch.nn as nn


class RNN(nn.Module):
    """Multi-layer Elman RNN with concatenated sequence projection."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        sequence_length: int,
        num_layers: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device,
            dtype=x.dtype,
        )
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        return self.fc(out)
