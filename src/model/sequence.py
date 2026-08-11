"""Shared head for LSTM/attention forecasters, with an optional per-step encoder."""

import torch.nn as nn

from model.blocks import LSTMAttentionBranch


class SequenceHeadNet(nn.Module):
    """`LSTMAttentionBranch` over a sequence of per-step feature vectors,
    with a LayerNorm->Dropout->Linear->GELU->Dropout->Linear(1) head,
    squeezed to (B,). `encoder`, if given, embeds each raw per-step input
    (B, T, ...) into a `feat_dim`-wide vector before the LSTM branch sees it
    -- e.g. a 1D CNN over a raw waveform, one embedding per hour."""

    def __init__(self, feat_dim, hidden=64, dropout=0.3, encoder=None):
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
        if self.encoder is not None:
            b, t = seq.shape[:2]
            seq = self.encoder(seq.reshape(b * t, *seq.shape[2:])).reshape(b, t, -1)
        return self.head(self.branch(seq)).squeeze(-1)
