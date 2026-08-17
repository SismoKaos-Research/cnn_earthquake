"""Dual-branch GRU/CNN forecaster: will an event follow this window?

A sequence of hours goes in; one logit comes out, for "a qualifying event
occurs within the horizon after the last hour of the window".

    catalog  (batch, seq_len, cat_dim)          -> GRU -> last hidden state
    waveform (batch, seq_len, 3, hour_samples)  -> CNN per hour -> GRU -> last
                                                -> concat -> MLP -> 1 logit

The waveform branch is optional. With `use_waveform=False` the model is a
catalog-only baseline and the second branch is not constructed at all, which
is the configuration to run first: it trains in minutes and establishes what
the waveform has to beat.

Run it through `gru_cnn_train.py`, which owns the data, the splits and the
baselines:

    # catalog only, the fast baseline
    python3 src/forecasting/gru_cnn_train.py \\
        --features combined_features_114d.parquet \\
        --catalog-path ../Sismokaos/data_downloader/catalogs/data_large.csv

    # add the waveform branch (needs a .f32 stream from sismokaos-cli)
    python3 src/forecasting/gru_cnn_train.py \\
        --features combined_features_114d.parquet \\
        --raw-f32 aegean_bodt_preprocessed.f32 \\
        --catalog-path ../Sismokaos/data_downloader/catalogs/data_large.csv

Read `--help` there for horizon, sequence length, folds and seeds.

**On the waveform branch's resolution.** The CNN ends in
`AdaptiveAvgPool1d(1)`, so each hour collapses to one vector and all
within-hour timing is discarded before the GRU sees it: the model can tell a
loud hour from a quiet one, not an impulsive arrival from a gradual rise.
That is deliberate -- it keeps the per-hour embedding to 64 numbers, so a
24-hour window stays tractable -- but it does mean this architecture cannot
express arrival shape. `--wave-pool` raises the pooled length if you want to
test whether that matters.
"""

import torch
import torch.nn as nn


class SeismicFusionModel(nn.Module):
    """Catalog GRU, optionally fused with a per-hour waveform CNN + GRU.

    Args:
        use_waveform: Build the waveform branch. When False the branch is not
            constructed, `forward` ignores its `wave_seq` argument, and the
            model is a catalog-only baseline.
        cat_dim: Catalog features per hour. Must match the feature count the
            loader supplies (3 for the RFE subset).
        wave_channels: Waveform components per hour, E/N/Z.
        cat_hidden: Catalog GRU hidden size.
        wave_embedding: Per-hour waveform embedding width, and the waveform
            GRU's hidden size.
        wave_pool: Pooled length per hour before flattening. 1 keeps one
            vector per hour; larger values retain coarse within-hour timing at
            a proportional cost in embedding width.
        dropout: Dropout in the classifier head.
    """

    def __init__(self, use_waveform=True, cat_dim=3, wave_channels=3,
                 cat_hidden=32, wave_embedding=64, wave_pool=1, dropout=0.3):
        super().__init__()
        self.use_waveform = use_waveform
        self.cat_hidden_size = cat_hidden
        self.wave_embedding_size = wave_embedding
        self.wave_pool = wave_pool

        self.catalog_gru = nn.GRU(
            input_size=cat_dim, hidden_size=cat_hidden, batch_first=True)

        if use_waveform:
            # Strided convolutions rather than pooling alone: an hour is tens
            # of thousands of samples, and reducing early keeps the per-hour
            # pass cheap enough to run seq_len of them per sample.
            self.cnn_extractor = nn.Sequential(
                nn.Conv1d(wave_channels, 16, kernel_size=15, stride=5),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(16, 32, kernel_size=9, stride=3),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(32, wave_embedding, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(wave_pool),
                nn.Flatten(),
            )
            self.wave_gru = nn.GRU(
                input_size=wave_embedding * wave_pool,
                hidden_size=wave_embedding,
                batch_first=True)
            fc_input_dim = cat_hidden + wave_embedding
        else:
            fc_input_dim = cat_hidden

        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, cat_seq, wave_seq=None):
        """Scores a batch of windows.

        Args:
            cat_seq: `(batch, seq_len, cat_dim)` catalog features.
            wave_seq: `(batch, seq_len, 3, hour_samples)` waveform. Ignored
                when `use_waveform` is False; an empty tensor is accepted
                there, since the default collate needs a tensor rather than
                None.

        Returns:
            `(batch, 1)` raw logits, for `BCEWithLogitsLoss`.

        Raises:
            ValueError: If the waveform branch is active but no waveform
                arrives, or its shape does not match the catalog sequence.
        """
        batch_size, seq_len, _ = cat_seq.shape

        cat_out, _ = self.catalog_gru(cat_seq)
        cat_final = cat_out[:, -1, :]

        if not self.use_waveform:
            return self.classifier(cat_final)

        if wave_seq is None or wave_seq.numel() == 0:
            raise ValueError(
                "use_waveform=True but no waveform arrived. Pass --raw-f32 so the "
                "loader has a stream to read, or build the model with "
                "use_waveform=False.")
        if wave_seq.shape[:2] != (batch_size, seq_len):
            raise ValueError(
                f"waveform is {tuple(wave_seq.shape[:2])} but the catalog sequence "
                f"is {(batch_size, seq_len)}; the two must describe the same hours.")

        # Every hour in the batch goes through the CNN as one flat batch, then
        # is folded back into sequences for the GRU.
        channels, samples = wave_seq.shape[2], wave_seq.shape[3]
        wave_flat = wave_seq.reshape(batch_size * seq_len, channels, samples)
        cnn_embeds = self.cnn_extractor(wave_flat)
        wave_embed_seq = cnn_embeds.view(batch_size, seq_len, -1)

        wave_out, _ = self.wave_gru(wave_embed_seq)
        wave_final = wave_out[:, -1, :]

        return self.classifier(torch.cat([cat_final, wave_final], dim=1))

    def n_params(self):
        """Trainable parameter count, for the params-per-sample line."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
