"""SE-ResNet trunk shared by the RAM/spectrogram classifiers and regressors.

Not a runnable script -- imported only. Callers: training.py
(ImprovedSeismicCNN subclasses this), cnn_regression.py
(RegressionSeismicCNN), cnn_ram_aux.py (RamAuxCNN).
"""

import torch
import torch.nn as nn

from seismolib.model.blocks import ResBlock


class SETrunk2D(nn.Module):
    """SE-ResNet trunk with global average pooling, optionally concatenating
    auxiliary scalars before a `classifier` head of width `num_classes`.

    `num_stages=4` keeps `layer1..layer4` as the trunk's state-dict keys
    regardless of caller, matching every existing checkpoint in this repo.
    """

    aux_dim = 0  # class-level default: full-object pickles saved before this
                 # attribute existed restore their __dict__ verbatim, without it

    def __init__(self, num_stages=4, in_channels=3, aux_dim=0, num_classes=1,
                dropout1=0.5, dropout2=0.3, hidden_dim=64):
        """Initializes the trunk and classifier head.

        Args:
            num_stages: 3 or 4 residual stages. 4 keeps `layer1..layer4` as
                the state-dict keys (matches existing checkpoints); 3 drops
                `layer4` to an identity and halves the final channel count,
                for a smaller model on short/low-signal inputs.
            in_channels: Number of input image channels.
            aux_dim: Width of an auxiliary scalar vector concatenated onto
                the pooled features before the classifier head. 0 disables
                the aux path entirely (forward's `aux` argument is ignored).
            num_classes: Width of the final linear layer -- 1 for binary
                classification/regression, N for N-way classification.
            dropout1: Dropout before the classifier's hidden layer.
            dropout2: Dropout before the classifier's output layer.
            hidden_dim: Width of the classifier's hidden layer.
        """
        super().__init__()
        self.aux_dim = aux_dim
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU()
        )
        self.layer1 = ResBlock(16, 32, stride=2)
        self.layer2 = ResBlock(32, 64, stride=2)
        self.layer3 = ResBlock(64, 128, stride=2)
        if num_stages >= 4:
            self.layer4 = ResBlock(128, 256, stride=2)
            final_channels = 256
        else:
            self.layer4 = nn.Identity()
            final_channels = 128
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(final_channels + aux_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, aux=None):
        """Runs the trunk and classifier head.

        Args:
            x: Input image batch, shape (batch, in_channels, height, width).
            aux: Auxiliary scalar batch, shape (batch, aux_dim). Ignored if
                `aux_dim` is 0 (including on legacy checkpoints loaded before
                this attribute existed, via the class-level default above).

        Returns:
            Tensor of shape (batch, num_classes) -- raw logits, no
            activation applied.
        """
        x = self.in_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.global_pool(x), 1)
        if self.aux_dim:
            x = torch.cat([x, aux], dim=1)
        return self.classifier(x)
