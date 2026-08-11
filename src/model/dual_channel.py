"""Dual-channel (1D LSTM+attention / 2D CNN / optional aux-scalar) fusion models."""

import torch
import torch.nn as nn

from model.blocks import CNNBranch, GatedFusion, LSTMAttentionBranch


class DualChannelTrunk(nn.Module):
    """Builds the 1D/2D/aux branches and fuses them into one vector of width
    `fused_dim`. Shared by every dual-channel model in this repo; subclasses
    attach whatever head(s) their task needs.

    `channels` ablates which branches are active: "all", "1d", "2d", "aux",
    "1d+aux", "2d+aux" (aux-only variants require `aux_dim > 0`).
    `fusion="linear"` is a*F1+b*F2 with learned scalars (the paper's
    default); `fusion="gate"` is a per-example gate (`model.blocks.GatedFusion`),
    only meaningful when both 1D and 2D are active.
    """

    def __init__(self, seq_dim, img_channels, aux_dim=0, hidden=64, fusion_dim=128,
                dropout=0.3, channels="all", fusion="linear",
                lstm_layers=1, lstm_heads=4):
        super().__init__()
        self.channels = channels
        self.fusion = fusion
        self.aux_dim = aux_dim
        self.use_1d = channels in ("all", "1d", "1d+aux")
        self.use_2d = channels in ("all", "2d", "2d+aux")
        self.use_aux = aux_dim > 0 and channels in ("all", "aux", "1d+aux", "2d+aux")
        if not (self.use_1d or self.use_2d or self.use_aux):
            raise ValueError(f"--channels {channels} disables every branch")
        if fusion not in ("linear", "gate"):
            raise ValueError(f"--fusion must be 'linear' or 'gate', got {fusion!r}")

        if self.use_1d:
            self.b1 = LSTMAttentionBranch(seq_dim, hidden=hidden, layers=lstm_layers,
                                          heads=lstm_heads, dropout=dropout)
            self.p1 = nn.Linear(self.b1.out_dim, fusion_dim)
        if self.use_2d:
            self.b2 = CNNBranch(img_channels, dropout=dropout)
            self.p2 = nn.Linear(self.b2.out_dim, fusion_dim)

        self.both = self.use_1d and self.use_2d
        if self.both and fusion == "gate":
            self.gated_fusion = GatedFusion(fusion_dim)
        else:
            # Learned fusion weights (a, b in the paper's notation). Also
            # used, harmlessly, as a global rescale in single-branch
            # ablations -- the optimizer settles it near 1 since there is
            # nothing to balance it against.
            self.w1 = nn.Parameter(torch.tensor(1.0))
            self.w2 = nn.Parameter(torch.tensor(1.0))

        self.fused_dim = (fusion_dim if (self.use_1d or self.use_2d) else 0) + \
                         (aux_dim if self.use_aux else 0)
        self.last_gate = None

    def _fuse(self, seq, img, aux):
        self.last_gate = None
        feats = []
        fused = None
        if self.both:
            f1 = self.p1(self.b1(seq))
            f2 = self.p2(self.b2(img))
            if self.fusion == "gate":
                fused, self.last_gate = self.gated_fusion(f1, f2)
            else:
                fused = self.w1 * f1 + self.w2 * f2
        elif self.use_1d:
            fused = self.w1 * self.p1(self.b1(seq))
        elif self.use_2d:
            fused = self.w2 * self.p2(self.b2(img))
        if fused is not None:
            feats.append(fused)
        if self.use_aux:
            feats.append(aux)
        return torch.cat(feats, dim=1)


class DualChannelNet(DualChannelTrunk):
    """`DualChannelTrunk` plus a single Sequential head of width `n_classes`
    (1 for binary/regression, 3+ for multiclass). `squeeze_output` matches
    each caller's existing convention for a single-logit head."""

    def __init__(self, seq_dim, img_channels, aux_dim=0, hidden=64, fusion_dim=128,
                dropout=0.3, channels="all", fusion="linear",
                lstm_layers=1, lstm_heads=4, n_classes=1, squeeze_output=False):
        super().__init__(seq_dim, img_channels, aux_dim=aux_dim, hidden=hidden,
                         fusion_dim=fusion_dim, dropout=dropout, channels=channels,
                         fusion=fusion, lstm_layers=lstm_layers, lstm_heads=lstm_heads)
        self.squeeze_output = squeeze_output
        self.head = nn.Sequential(
            nn.LayerNorm(self.fused_dim),
            nn.Dropout(dropout),
            nn.Linear(self.fused_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, n_classes),
        )

    def forward(self, seq, img, aux=None):
        out = self.head(self._fuse(seq, img, aux))
        return out.squeeze(-1) if self.squeeze_output else out


class DualChannelDualHeadNet(DualChannelTrunk):
    """`DualChannelTrunk` with a shared post-fusion trunk feeding two heads
    -- e.g. `cnn_lstm_forecast.py`'s binary "will it happen" + regression
    "how big" pair. Both outputs are squeezed to (B,)."""

    def __init__(self, seq_dim, img_channels, aux_dim=0, hidden=64, fusion_dim=128,
                dropout=0.3, channels="all", lstm_layers=1, lstm_heads=4):
        super().__init__(seq_dim, img_channels, aux_dim=aux_dim, hidden=hidden,
                         fusion_dim=fusion_dim, dropout=dropout, channels=channels,
                         fusion="linear", lstm_layers=lstm_layers, lstm_heads=lstm_heads)
        self.trunk = nn.Sequential(
            nn.LayerNorm(self.fused_dim),
            nn.Dropout(dropout),
            nn.Linear(self.fused_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.binary_out = nn.Linear(fusion_dim, 1)
        self.magnitude_out = nn.Linear(fusion_dim, 1)

    def forward(self, seq, img, aux=None):
        trunk_out = self.trunk(self._fuse(seq, img, aux))
        return self.binary_out(trunk_out).squeeze(-1), self.magnitude_out(trunk_out).squeeze(-1)
