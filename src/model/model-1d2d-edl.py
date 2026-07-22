import torch
import torch.nn as nn
import torch.nn.functional as F


class Channel1D(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_heads=4, num_layers=2):
        super(Channel1D, self).__init__()
        
        # LSTM layer to capture long-term temporal dependencies
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True # Bidirectional helps capture wave context
        )
        
        # Multi-Head Self-Attention (MSA)
        # Hidden size is multiplied by 2 because of the bidirectional LSTM
        self.msa = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=num_heads, batch_first=True)
        
    def forward(self, x):
        # x shape: (Batch, Sequence_Length, Channels) -> e.g., (B, 6000, 3)
        lstm_out, _ = self.lstm(x)
        
        # MSA expects (Batch, Seq, Feature) when batch_first=True
        attn_out, _ = self.msa(lstm_out, lstm_out, lstm_out)
        
        # Pool the sequence dimension to get a single feature vector per batch item
        # Global average pooling over the sequence length
        pooled_out = torch.mean(attn_out, dim=1) 
        
        return pooled_out


class Channel2D(nn.Module):
    def __init__(self, in_channels=3):
        super(Channel2D, self).__init__()
        
        # 3 Convolutional blocks matching the paper's structure
        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width) -> e.g., (B, 3, 94, 94)
        x = self.conv_block(x)
        x = self.adaptive_pool(x)
        
        # Flatten for the fully connected fusion layer
        x = torch.flatten(x, 1)
        return x


class EDL1D2D(nn.Module):
    def __init__(self, num_classes=3):
        """
        num_classes: e.g., 3 for [Noise, P-Wave, S-Wave] classification
        """
        super(EDL1D2D, self).__init__()
        
        self.channel_1d = Channel1D(input_size=3, hidden_size=64, num_heads=4)
        self.channel_2d = Channel2D(in_channels=3)
        
        # Calculate feature sizes
        # 1D out: 64 hidden_size * 2 (bidirectional) = 128
        # 2D out: 128 channels * 4 * 4 (from adaptive pool) = 2048
        combined_features = 128 + 2048
        
        # Feature Fusion & Fully Connected Classifier
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # Helps prevent overfitting
            nn.Linear(512, num_classes)
        )
        
    def forward(self, raw_1d, img_2d):
        # Extract features from both channels simultaneously 
        feat_1d = self.channel_1d(raw_1d)
        feat_2d = self.channel_2d(img_2d)
        
        # Feature Fusion: Concatenate along the feature dimension
        fused_features = torch.cat((feat_1d, feat_2d), dim=1)
        
        # Classification
        out = self.classifier(fused_features)
        
        return out
