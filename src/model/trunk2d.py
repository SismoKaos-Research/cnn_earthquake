"""SE-ResNet trunk shared by the RAM/spectrogram classifiers and regressors."""

import torch
import torch.nn as nn

from model.blocks import ResBlock


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
        x = self.in_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.global_pool(x), 1)
        if self.aux_dim:
            x = torch.cat([x, aux], dim=1)
        return self.classifier(x)
