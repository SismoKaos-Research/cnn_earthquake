"""Temporal convolutional network for sequence forecasting.

Not a runnable script -- imported only. Callers: feature_gru_tcn.py (which
re-exports `Chomp1d`, `TemporalBlock` and `ForecastTCN`), sismokaos.model.registry
(as `tcn`).

Moved out of `feature_gru_tcn.py` unchanged, for the reason given in
`sismokaos/model/recurrent.py`: a model reachable only by importing its trainer
is a model the registry cannot list. Class bodies are byte-identical, so state
dicts written before the move load after it.

The registry exposes `levels` and `hidden` rather than `num_channels`, because a
list-valued flag is awkward on a command line and every use in this repo passed
the same width at every level anyway (`[--hidden] * 3`). Passing `num_channels`
directly still works for callers constructing the class themselves.
"""

import torch.nn as nn


class Chomp1d(nn.Module):
    """Removes padding from the end of a sequence for causal 1D convolutions."""
    def __init__(self, chomp_size):
        """Stores how many trailing samples the convolution's padding added.

        Args:
            chomp_size: Number of trailing samples to drop.
        """
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """Drops the trailing padding.

        Args:
            x: Input, shape (batch, channels, time).

        Returns:
            Tensor of shape (batch, channels, time - chomp_size).
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """A single residual block for the TCN."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.3):
        """Initializes two dilated causal convolutions and the residual path.

        Args:
            n_inputs: Input channels.
            n_outputs: Output channels.
            kernel_size: Convolution kernel width.
            stride: Convolution stride.
            dilation: Dilation factor for both convolutions.
            padding: Left padding, chomped back off to keep the block causal.
            dropout: Dropout after each convolution.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        """Applies both convolutions and adds the (optionally projected) input.

        Args:
            x: Input, shape (batch, n_inputs, time).

        Returns:
            Tensor of shape (batch, n_outputs, time).
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class ForecastTCN(nn.Module):
    """Temporal Convolutional Network for sequence forecasting."""
    def __init__(self, feat_dim, num_channels=[64, 64, 64], kernel_size=3, dropout=0.3):
        """Initializes the dilated block stack and single-logit head.

        Args:
            feat_dim: Per-step feature width.
            num_channels: Output channels per level; its length is the number
                of levels, and dilation doubles at each one.
            kernel_size: Convolution kernel width in every block.
            dropout: Dropout inside the blocks and the head.
        """
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = feat_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                        dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                                        dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, 1)
        )

    def forward(self, x):
        """Scores a batch of sequences from the last timestep's features.

        Args:
            x: Input sequence, shape (batch, time, feat_dim).

        Returns:
            Tensor of shape (batch,) -- one raw logit per sequence.
        """
        x = x.transpose(1, 2)
        out = self.tcn(x)
        out = out[:, :, -1]
        return self.head(out).squeeze(-1)
