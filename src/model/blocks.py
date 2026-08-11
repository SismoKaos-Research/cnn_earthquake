"""Building blocks shared by the SE-ResNet trunk and dual-channel models."""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    """A standard Residual Block with an integrated SE Block."""
    def __init__(self, in_channels, out_channels, stride=1):
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
        out = torch.nn.functional.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return torch.nn.functional.gelu(out)


class LSTMAttentionBranch(nn.Module):
    """LSTM for long-range order, then multi-head self-attention to weight steps."""

    def __init__(self, in_dim, hidden=64, layers=1, heads=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        d = hidden * 2
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.out_dim = d

    def forward(self, x):
        h, _ = self.lstm(x)
        a, _ = self.attn(h, h, h)
        h = self.norm(h + a)             # residual, as in the transformer block
        return h.mean(dim=1)             # pool over time


class CNNBranch(nn.Module):
    """Compact CNN over the RAM image. The images are small (32x32 by default),
    so a 4-stage ResNet would be heavily over-provisioned here."""

    def __init__(self, in_channels=3, width=32, dropout=0.2):
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
        return torch.flatten(self.net(x), 1)


class GatedFusion(nn.Module):
    """
    Per-example gate deciding how much to trust each branch, replacing a
    fixed pair of scalars (a*F1 + b*F2, same for every example) with
    g(x)*F1 + (1-g(x))*F2, where g = sigmoid(MLP([F1, F2])) is conditioned on
    both branches' own features for THIS example.
    """

    def __init__(self, dim, hidden=None, dropout=0.1):
        super().__init__()
        hidden = hidden or max(8, dim // 2)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, f1, f2):
        g = torch.sigmoid(self.net(torch.cat([f1, f2], dim=1)))
        return g * f1 + (1.0 - g) * f2, g
