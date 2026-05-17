import torch
import torch.nn as nn


class LinearBaseline(nn.Module):
    """
    Flat linear baseline: flattens the input window and predicts future glucose
    with a single linear layer. Useful as a lower-bound comparison.
    """

    def __init__(self, seq_len: int = 24, input_size: int = 1, num_horizons: int = 2):
        super().__init__()
        self.fc = nn.Linear(seq_len * input_size, num_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        return self.fc(x.flatten(1))    # (batch, num_horizons)


class NaiveLastValueBaseline(nn.Module):
    """
    Zero-order hold: predicts that glucose stays constant at its last observed value.
    Sets the floor — any trained model should beat this.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1, 0:1]           # (batch, 1)
        return last.expand(-1, 2)       # repeat for both horizons
