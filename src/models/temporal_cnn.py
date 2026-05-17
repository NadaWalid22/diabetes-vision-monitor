import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class _CausalConv1d(nn.Module):
    """Causal (non-leaking) 1D convolution with dilation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        # Padding on the left only so future timesteps don't leak into past
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            _CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # 1x1 conv to match channel dims for the skip connection
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.skip(x))


class GlucoseTemporalCNN(nn.Module):
    """
    Temporal Convolutional Network for glucose prediction.

    Uses stacked dilated causal residual blocks so the receptive field grows
    exponentially with depth while preserving strict causality.
    """

    def __init__(
        self,
        input_size: int = 1,
        num_channels: list = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_horizons: int = 2,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [32, 64, 64]

        layers = []
        in_ch = input_size
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            layers.append(_ResidualBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size) → transpose for Conv1d
        x = x.permute(0, 2, 1)          # (batch, input_size, seq_len)
        x = self.network(x)              # (batch, channels, seq_len)
        x = x[:, :, -1]                  # last timestep
        return self.fc(x)                # (batch, num_horizons)
