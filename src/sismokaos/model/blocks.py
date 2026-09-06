"""Building blocks shared by the SE-ResNet trunk (model/trunk2d.py) and the
dual-channel / sequence models (model/dual_channel.py, model/sequence.py).

Not a runnable script -- imported only. Callers: training.py (ResBlock,
SEBlock), cnn_lstm.py (LSTMAttentionBranch, re-exported for
cnn_groundmotion.py, feature_lstm_forecast.py, raw_cnn_lstm_forecast.py),
model/trunk2d.py (ResBlock), model/dual_channel.py (CNNBranch, GatedFusion,
LSTMAttentionBranch), model/sequence.py (LSTMAttentionBranch).
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-excitation gate: pools each channel to a scalar, learns a
    per-channel reweighting from those scalars, and rescales the input by it.
    """

    def __init__(self, channels, reduction=16):
        """Initializes the squeeze-and-excitation gate.

        Args:
            channels: Number of input/output channels.
            reduction: Bottleneck ratio for the excitation MLP; the hidden
                width is ``max(1, channels // reduction)``.
        """
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Rescales each channel of ``x`` by its learned gate value.

        Args:
            x: Input feature map, shape (batch, channels, height, width).

        Returns:
            Tensor of the same shape as ``x``.
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    """A standard residual block with an integrated SE block."""

    def __init__(self, in_channels, out_channels, stride=1):
        """Initializes the residual block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride of the first conv (and of the projection shortcut,
                if one is needed). 1 keeps spatial size; >1 downsamples.
        """
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """Applies the block: conv-bn-gelu x2, SE gate, residual add, gelu.

        Args:
            x: Input feature map, shape (batch, in_channels, height, width).

        Returns:
            Tensor of shape (batch, out_channels, height', width'), where
            height'/width' are downsampled by ``stride`` if it was >1.
        """
        out = torch.nn.functional.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return torch.nn.functional.gelu(out)


class LSTMAttentionBranch(nn.Module):
    """LSTM for long-range order, then multi-head self-attention to weight steps."""

    def __init__(self, in_dim, hidden=64, layers=1, heads=4, dropout=0.2):
        """Initializes the LSTM+attention branch.

        Args:
            in_dim: Size of each step's input feature vector.
            hidden: LSTM hidden size per direction; the bidirectional output
                (and attention/LayerNorm width) is ``hidden * 2``.
            layers: Number of stacked LSTM layers.
            heads: Number of attention heads. Must divide ``hidden * 2``.
            dropout: Dropout used inside the LSTM (when ``layers > 1``) and
                the attention module.
        """
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        d = hidden * 2
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.out_dim = d

    def forward(self, x):
        """Encodes a sequence into one pooled embedding.

        Args:
            x: Input sequence, shape (batch, time, in_dim).

        Returns:
            Tensor of shape (batch, out_dim), the time-mean of the
            attention+residual output.
        """
        h, _ = self.lstm(x)
        a, _ = self.attn(h, h, h)
        h = self.norm(h + a)             # residual, as in the transformer block
        return h.mean(dim=1)             # pool over time


class ConvSeqBranch(nn.Module):
    """1D CNN over the raw waveform, optionally followed by BiLSTM + attention.

    `LSTMAttentionBranch` feeds 600 raw 100 Hz samples straight into an LSTM,
    so the only local structure available to it is whatever recurrence can
    accumulate one 10 ms sample at a time. The detectors this project compares
    itself against do the opposite: EQTransformer is CNN -> BiLSTM -> attention
    and PhaseNet is a U-Net of 1D convolutions, both extracting local waveform
    features convolutionally before any recurrence.

    This branch adds that missing front end. Strided convolutions reduce the
    sequence roughly 8x before the recurrent layer, which also makes the
    self-attention that follows ~64x cheaper (its cost is quadratic in
    sequence length).

    `use_lstm=False` stops after the convolutions and mean-pools, isolating
    whether the recurrence contributes anything once local features exist.

    Args:
        in_dim: Channels per timestep of the input sequence (3 for Z/N/E).
        hidden: LSTM hidden size per direction. Output width is ``hidden * 2``
            with an LSTM, or ``conv_width`` without one.
        layers: Stacked LSTM layers.
        heads: Attention heads. Must divide ``hidden * 2``.
        dropout: Dropout after the convolution stages and inside attention.
        conv_width: Channel width of the final convolution stage.
    """

    def __init__(self, in_dim, hidden=64, layers=1, heads=4, dropout=0.2,
                 use_lstm=True, conv_width=96):
        super().__init__()
        self.use_lstm = use_lstm

        def stage(cin, cout, k, s):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=k, stride=s, padding=k // 2,
                          bias=False),
                nn.BatchNorm1d(cout), nn.GELU())

        # 600 -> 300 -> 150 -> 75 at 100 Hz. Kernels stay wide enough at the
        # first stage (70 ms) to see an arrival's onset rather than one cycle.
        self.conv = nn.Sequential(
            stage(in_dim, conv_width // 4, 7, 2),
            stage(conv_width // 4, conv_width // 2, 5, 2),
            nn.Dropout(dropout),
            stage(conv_width // 2, conv_width, 5, 2),
        )

        if use_lstm:
            self.lstm = nn.LSTM(conv_width, hidden, num_layers=layers,
                                batch_first=True, bidirectional=True,
                                dropout=dropout if layers > 1 else 0.0)
            d = hidden * 2
            self.attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                              batch_first=True)
            self.norm = nn.LayerNorm(d)
            self.out_dim = d
        else:
            self.out_dim = conv_width

    def forward(self, x):
        """Encodes a raw waveform sequence into one pooled embedding.

        Args:
            x: Input sequence, shape (batch, time, in_dim).

        Returns:
            Tensor of shape (batch, out_dim), mean-pooled over time.
        """
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)   # (B, T', conv_width)
        if not self.use_lstm:
            return h.mean(dim=1)
        h, _ = self.lstm(h)
        a, _ = self.attn(h, h, h)
        h = self.norm(h + a)
        return h.mean(dim=1)


class CNNBranch(nn.Module):
    """Compact CNN over the RAM image. The images are small (32x32 by default),
    so a 4-stage ResNet would be heavily over-provisioned here."""

    def __init__(self, in_channels=3, width=32, dropout=0.2):
        """Initializes the compact image branch.

        Args:
            in_channels: Number of input image channels.
            width: Base channel width; the three conv stages use
                ``width``, ``width*2``, ``width*4`` channels.
            dropout: Dropout2d probability applied after the second stage.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2), nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 4), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = width * 4

    def forward(self, x):
        """Encodes an image into one pooled embedding.

        Args:
            x: Input image batch, shape (batch, in_channels, height, width).

        Returns:
            Tensor of shape (batch, out_dim).
        """
        return torch.flatten(self.net(x), 1)


class GatedFusion(nn.Module):
    """
    Per-example gate deciding how much to trust each branch, replacing a
    fixed pair of scalars (a*F1 + b*F2, same for every example) with
    g(x)*F1 + (1-g(x))*F2, where g = sigmoid(MLP([F1, F2])) is conditioned on
    both branches' own features for THIS example.
    """

    def __init__(self, dim, hidden=None, dropout=0.1):
        """Initializes the gated fusion module.

        Args:
            dim: Width of each of the two branch embeddings being fused
                (they must match).
            hidden: Hidden width of the gating MLP. Defaults to
                ``max(8, dim // 2)`` when None.
            dropout: Dropout probability inside the gating MLP.
        """
        super().__init__()
        hidden = hidden or max(8, dim // 2)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, f1, f2):
        """Fuses two branch embeddings with a learned per-example gate.

        Args:
            f1: First branch's embedding, shape (batch, dim).
            f2: Second branch's embedding, shape (batch, dim).

        Returns:
            Tuple of (fused, gate): ``fused`` has shape (batch, dim);
            ``gate`` has shape (batch, 1) and is in (0, 1), where values
            near 1 favor ``f1`` and values near 0 favor ``f2``.
        """
        g = torch.sigmoid(self.net(torch.cat([f1, f2], dim=1)))
        return g * f1 + (1.0 - g) * f2, g
