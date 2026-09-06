"""Dual-channel (1D LSTM+attention / 2D CNN / optional aux-scalar) fusion models.

Not a runnable script -- imported only. Callers: cnn_lstm.py
(DualChannelRiskNet subclasses DualChannelNet), cnn_lstm_classify.py
(DualChannelBinaryNet), cnn_lstm_classify_aux.py (DualChannelAuxBinaryNet),
cnn_lstm_regression.py (DualChannelRegressionNet), cnn_lstm_forecast.py
(DualChannelForecastNet subclasses DualChannelDualHeadNet).
"""

import torch
import torch.nn as nn

from sismokaos.model.blocks import (CNNBranch, ConvSeqBranch, GatedFusion,
                                    LSTMAttentionBranch)


class DualChannelTrunk(nn.Module):
    """Builds the 1D/2D/aux branches and fuses them into one vector of width
    `fused_dim`. Shared by every dual-channel model in this repo; subclasses
    attach whatever head(s) their task needs.

    `channels` ablates which branches are active: "all", "1d", "2d", "aux",
    "1d+aux", "2d+aux" (aux-only variants require `aux_dim > 0`), and "1d+2d"
    (both waveform branches with the aux vector withheld).
    `fusion="linear"` is a*F1+b*F2 with learned scalars (the paper's
    default); `fusion="gate"` is a per-example gate (`model.blocks.GatedFusion`),
    only meaningful when both 1D and 2D are active.
    """

    def __init__(self, seq_dim, img_channels, aux_dim=0, hidden=64, fusion_dim=128,
                dropout=0.3, channels="all", fusion="linear",
                lstm_layers=1, lstm_heads=4, branch1d="lstm"):
        """Initializes the 1D/2D/aux branches and fusion.

        Args:
            seq_dim: Per-step feature width of the 1D sequence input.
            img_channels: Number of channels of the 2D image input.
            aux_dim: Width of an auxiliary scalar vector concatenated after
                fusion. 0 disables the aux branch.
            hidden: LSTM hidden size (per direction) for the 1D branch.
            fusion_dim: Common width both branches are projected to before
                fusion, and the fused output's width.
            dropout: Dropout used throughout the branches and fusion.
            channels: Which branches are active -- "all", "1d", "2d", "aux",
                "1d+aux", "2d+aux" (aux-only variants require `aux_dim > 0`),
                or "1d+2d" (both waveform branches, no aux -- what a cascade
                can actually supply at run time).
            fusion: "linear" (a*F1+b*F2 with learned scalars) or "gate" (a
                per-example gate, `model.blocks.GatedFusion`); "gate" only
                takes effect when both 1D and 2D are active.
            lstm_layers: Number of stacked LSTM layers in the 1D branch.
            lstm_heads: Number of attention heads in the 1D branch.
            branch1d: Architecture of the 1D branch. "lstm" (default) is
                `LSTMAttentionBranch`, which reads raw samples directly and is
                what every existing result used. "cnn-lstm" prepends a strided
                1D convolutional encoder (`ConvSeqBranch`), matching the
                CNN-then-recurrence order EQTransformer and PhaseNet use;
                "cnn" keeps the convolutions and drops the recurrence.

        Raises:
            ValueError: If `channels` disables every branch, or `fusion` is
                not "linear" or "gate".
        """
        super().__init__()
        self.channels = channels
        self.fusion = fusion
        self.aux_dim = aux_dim
        self.use_1d = channels in ("all", "1d", "1d+aux", "1d+2d")
        self.use_2d = channels in ("all", "2d", "2d+aux", "1d+2d")
        # "1d+2d" is deliberately absent here: both waveform branches, no aux.
        # It exists because a deployable cascade cannot supply the aux vector.
        # `log_distance` is the epicentral distance to a CATALOGUED hypocentre,
        # and a window the detector just flagged has no catalogue entry, so the
        # one configuration an operational stage 2 needs -- everything the
        # waveform gives and nothing it does not -- had no name.
        self.use_aux = aux_dim > 0 and channels in ("all", "aux", "1d+aux", "2d+aux")
        if not (self.use_1d or self.use_2d or self.use_aux):
            raise ValueError(f"--channels {channels} disables every branch")
        if fusion not in ("linear", "gate"):
            raise ValueError(f"--fusion must be 'linear' or 'gate', got {fusion!r}")

        if branch1d not in ("lstm", "cnn", "cnn-lstm"):
            raise ValueError(
                f"branch1d must be 'lstm', 'cnn' or 'cnn-lstm', got {branch1d!r}")
        self.branch1d = branch1d

        if self.use_1d:
            if branch1d == "lstm":
                self.b1 = LSTMAttentionBranch(seq_dim, hidden=hidden,
                                              layers=lstm_layers,
                                              heads=lstm_heads, dropout=dropout)
            else:
                self.b1 = ConvSeqBranch(seq_dim, hidden=hidden, layers=lstm_layers,
                                        heads=lstm_heads, dropout=dropout,
                                        use_lstm=(branch1d == "cnn-lstm"))
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
        """Runs the active branches and fuses them into one vector.

        Args:
            seq: 1D sequence input, shape (batch, time, seq_dim). Unused if
                `use_1d` is False.
            img: 2D image input, shape (batch, img_channels, height, width).
                Unused if `use_2d` is False.
            aux: Auxiliary scalar input, shape (batch, aux_dim). Unused if
                `use_aux` is False.

        Returns:
            Tensor of shape (batch, fused_dim). Also sets `self.last_gate`
            to the per-example gate (batch, 1) when `fusion="gate"` and both
            branches are active, else None.
        """
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
                lstm_layers=1, lstm_heads=4, n_classes=1, squeeze_output=False,
                branch1d="lstm"):
        """Initializes the trunk (see `DualChannelTrunk.__init__`) plus a head.

        Args:
            seq_dim: See `DualChannelTrunk.__init__`.
            img_channels: See `DualChannelTrunk.__init__`.
            aux_dim: See `DualChannelTrunk.__init__`.
            hidden: See `DualChannelTrunk.__init__`.
            fusion_dim: See `DualChannelTrunk.__init__`.
            dropout: See `DualChannelTrunk.__init__`.
            channels: See `DualChannelTrunk.__init__`.
            fusion: See `DualChannelTrunk.__init__`.
            lstm_layers: See `DualChannelTrunk.__init__`.
            lstm_heads: See `DualChannelTrunk.__init__`.
            n_classes: Width of the head's output layer -- 1 for
                binary/regression, 3+ for multiclass.
            squeeze_output: If True, squeezes the last dimension off the
                output (for a single-logit head returned as shape (batch,)
                rather than (batch, 1)).
        """
        super().__init__(seq_dim, img_channels, aux_dim=aux_dim, hidden=hidden,
                         fusion_dim=fusion_dim, dropout=dropout, channels=channels,
                         fusion=fusion, lstm_layers=lstm_layers, lstm_heads=lstm_heads,
                         branch1d=branch1d)
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
        """Fuses the branches and applies the head.

        Args:
            seq: See `DualChannelTrunk._fuse`.
            img: See `DualChannelTrunk._fuse`.
            aux: See `DualChannelTrunk._fuse`.

        Returns:
            Tensor of shape (batch, n_classes), or (batch,) if
            `squeeze_output` and `n_classes == 1`.
        """
        out = self.head(self._fuse(seq, img, aux))
        return out.squeeze(-1) if self.squeeze_output else out


class DualChannelDualHeadNet(DualChannelTrunk):
    """`DualChannelTrunk` with a shared post-fusion trunk feeding two heads
    -- e.g. `cnn_lstm_forecast.py`'s binary "will it happen" + regression
    "how big" pair. Both outputs are squeezed to (B,)."""

    def __init__(self, seq_dim, img_channels, aux_dim=0, hidden=64, fusion_dim=128,
                dropout=0.3, channels="all", lstm_layers=1, lstm_heads=4,
                fusion="linear", branch1d="lstm"):
        """Initializes the trunk (see `DualChannelTrunk.__init__`) plus two heads.

        `fusion` and `branch1d` were fixed at the trunk's defaults here until
        the registry exposed them per task. Nothing about two heads constrains
        either one -- they belong to the trunk, which is the same trunk the
        single-head models use -- so the restriction was incidental. Both keep
        the trunk's defaults, so existing checkpoints and callers are unaffected.

        Args:
            seq_dim: See `DualChannelTrunk.__init__`.
            img_channels: See `DualChannelTrunk.__init__`.
            aux_dim: See `DualChannelTrunk.__init__`.
            hidden: See `DualChannelTrunk.__init__`.
            fusion_dim: See `DualChannelTrunk.__init__`.
            dropout: See `DualChannelTrunk.__init__`.
            channels: See `DualChannelTrunk.__init__`.
            lstm_layers: See `DualChannelTrunk.__init__`.
            lstm_heads: See `DualChannelTrunk.__init__`.
            fusion: See `DualChannelTrunk.__init__`.
            branch1d: See `DualChannelTrunk.__init__`.
        """
        super().__init__(seq_dim, img_channels, aux_dim=aux_dim, hidden=hidden,
                         fusion_dim=fusion_dim, dropout=dropout, channels=channels,
                         fusion=fusion, lstm_layers=lstm_layers, lstm_heads=lstm_heads,
                         branch1d=branch1d)
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
        """Fuses the branches and applies the shared trunk, then both heads.

        Args:
            seq: See `DualChannelTrunk._fuse`.
            img: See `DualChannelTrunk._fuse`.
            aux: See `DualChannelTrunk._fuse`.

        Returns:
            Tuple of (binary_logit, magnitude) tensors, each shape (batch,).
        """
        trunk_out = self.trunk(self._fuse(seq, img, aux))
        return self.binary_out(trunk_out).squeeze(-1), self.magnitude_out(trunk_out).squeeze(-1)
