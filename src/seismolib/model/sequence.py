"""Shared head for LSTM/attention forecasters, with an optional per-step encoder.

Not a runnable script -- imported only. Callers: feature_lstm_forecast.py
(ForecastLSTM subclasses this directly, no encoder), raw_cnn_lstm_forecast.py
(RawCNNLSTM, encoder=RawWaveformEncoder), raw100hz_cnn_lstm_forecast.py
(NativeCNNLSTM, encoder=NativeWaveformEncoder).
"""

import torch.nn as nn

from seismolib.model.blocks import LSTMAttentionBranch


class SequenceHeadNet(nn.Module):
    """`LSTMAttentionBranch` over a sequence of per-step feature vectors,
    with a LayerNorm->Dropout->Linear->GELU->Dropout->Linear(1) head,
    squeezed to (B,). `encoder`, if given, embeds each raw per-step input
    (B, T, ...) into a `feat_dim`-wide vector before the LSTM branch sees it
    -- e.g. a 1D CNN over a raw waveform, one embedding per hour."""

    def __init__(self, feat_dim, hidden=64, dropout=0.3, encoder=None):
        """Initializes the LSTM/attention branch and single-logit head.

        Args:
            feat_dim: Per-step feature width the LSTM branch consumes --
                either the raw input's width (`encoder=None`) or `encoder`'s
                output width.
            hidden: LSTM hidden size (per direction) and head hidden width.
            dropout: Dropout used throughout the branch and head.
            encoder: Optional per-step encoder module (e.g. a 1D CNN) that
                embeds each raw step into a `feat_dim`-wide vector before the
                LSTM branch sees it. None passes each step through as-is.
        """
        super().__init__()
        self.encoder = encoder
        self.branch = LSTMAttentionBranch(feat_dim, hidden=hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(self.branch.out_dim),
            nn.Dropout(dropout),
            nn.Linear(self.branch.out_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, seq):
        """Encodes (optionally) and forecasts from a sequence.

        Args:
            seq: Input sequence. Shape (batch, time, feat_dim) if
                `encoder` is None; shape (batch, time, ...) matching
                `encoder`'s expected per-step input otherwise, e.g.
                (batch, time, channels, samples) for a 1D CNN encoder.

        Returns:
            Tensor of shape (batch,) -- a single raw logit per sequence.
        """
        if self.encoder is not None:
            b, t = seq.shape[:2]
            seq = self.encoder(seq.reshape(b * t, *seq.shape[2:])).reshape(b, t, -1)
        return self.head(self.branch(seq)).squeeze(-1)
