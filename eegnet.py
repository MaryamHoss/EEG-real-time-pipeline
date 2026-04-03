"""EEGNet classifier (Lawhern et al.) for (batch, channels, time) MI decoding."""

from __future__ import annotations

import torch
from torch import nn


class EEGNet(nn.Module):
    """
    Compact CNN for EEG. Expects ``(batch, n_channels, n_times)`` internally reshaped to
    ``(batch, 1, n_channels, n_times)``.
    """

    def __init__(
        self,
        n_channels: int, # Number of EEG channels
        n_times: int, # Number of time points
        n_classes: int = 2, # Number of classes (left vs right hand)
        *,
        F1: int = 8, # Number of filters in the first convolutional layer
        D: int = 2, # Depth factor: how many convolutional layers to stack
        F2: int = 16, # Number of filters in the second convolutional layer
        kernel_length: int | None = None, # Length of the convolutional kernel
        dropout: float = 0.25, 
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        if kernel_length is None:
            kernel_length = max(8, min(128, n_times // 4))
        if kernel_length % 2 == 0:
            kernel_length += 1

        self.block_t = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(F1),
        )
        self.block_s = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block_sep = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        """Since the size of the data changes as it goes through the filters and pools, 
        it's hard to guess the final number of inputs for the last layer.
        This code creates zeros, runs it through the blocks, sees how many numbers come out,
         and then builds the fc (Fully Connected) layer to match that exact size."""
        with torch.no_grad():
            fake = torch.zeros(1, 1, n_channels, n_times)
            h = self.block_t(fake)
            h = self.block_s(h)
            h = self.block_sep(h)
            n_flat = h.numel()
        self.fc = nn.Linear(n_flat, n_classes) # Fully Connected layer to match the size of the data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B, C, T); got {tuple(x.shape)}")
        # x is (Batch, Channels, Time)
        x = x.unsqueeze(1) # Add a dimension to make it (Batch, 1, Channels, Time)
        x = self.block_t(x)
        x = self.block_s(x)
        x = self.block_sep(x)
        x = x.reshape(x.shape[0], -1) # Flatten the data to (Batch, Features)
        return self.fc(x) # Pass the data through the fully connected layer to get the logits
