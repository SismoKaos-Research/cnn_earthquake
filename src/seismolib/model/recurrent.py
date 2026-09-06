"""GRU sequence forecaster with additive attention.

Not a runnable script -- imported only. Callers: feature_gru_tcn.py (which
re-exports `ForecastGRU` for its existing `from forecasting.feature_gru_tcn
import ForecastGRU` callers), seismolib.model.registry (as `gru`).

Moved out of `feature_gru_tcn.py` unchanged. It lived inside a trainer, which
meant the only way to reach the architecture was to import the training script
-- and the registry cannot do that for a dozen models without importing a dozen
argparse-bearing modules to list them. The class body is byte-identical, so
state dicts written before the move load after it.
"""

import torch
import torch.nn as nn


class ForecastGRU(nn.Module):
    """GRU branch for sequence forecasting."""
    def __init__(self, feat_dim, hidden=64, dropout=0.3):
        """Initializes the GRU, attention pooling, and single-logit head.

        Args:
            feat_dim: Per-step feature width.
            hidden: GRU hidden size (single direction).
            dropout: Dropout inside the head. Not passed to the GRU -- see
                the comment below.
        """
        super().__init__()
        # Dropout set to 0 here because PyTorch GRU dropout only applies *between*
        # layers in a multi-layer stack. We use the head's dropout instead.
        self.gru = nn.GRU(feat_dim, hidden, batch_first=True, dropout=0)
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        """Scores a batch of sequences.

        Args:
            x: Input sequence, shape (batch, time, feat_dim).

        Returns:
            Tensor of shape (batch,) -- one raw logit per sequence.
        """
        out, _ = self.gru(x)
        attn_weights = torch.softmax(self.attn(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        return self.head(context).squeeze(-1)
