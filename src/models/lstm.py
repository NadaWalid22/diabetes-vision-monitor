import torch
import torch.nn as nn


class GlucoseLSTM(nn.Module):
    """
    LSTM for multi-horizon glucose prediction.

    Predicts glucose values at multiple future time steps (e.g. 30-min, 60-min)
    from a window of historical CGM readings.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_horizons: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Take the last timestep's output
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)  # (batch, num_horizons)
