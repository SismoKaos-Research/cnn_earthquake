import torch
import torch.nn as nn


class SeismicFusionModel(nn.Module):
    def __init__(self, use_waveform=True, cat_dim=3, seq_len=24, wave_channels=3):
        """
        Dual-branch Seismic Forecaster.
        
        Args:
            use_waveform (bool): If False, runs as a lightning-fast catalog-only baseline.
                                 If True, spins up the CNN feature extractor for raw data.
            cat_dim (int): Number of catalog features (3 from LightGBM RFE).
            seq_len (int): Trailing hours to look back (e.g., 24).
            wave_channels (int): E, N, Z (3).
        """
        super().__init__()
        self.use_waveform = use_waveform
        
        # Catalog Branch 
        self.cat_hidden_size = 32
        self.catalog_gru = nn.GRU(
            input_size=cat_dim, 
            hidden_size=self.cat_hidden_size, 
            batch_first=True
        )
        
        # Waveform Branch 
        if self.use_waveform:
            self.wave_embedding_size = 64
            self.cnn_extractor = nn.Sequential(
                nn.Conv1d(wave_channels, 16, kernel_size=15, stride=5),
                nn.ReLU(),
                nn.MaxPool1d(4),
                
                nn.Conv1d(16, 32, kernel_size=9, stride=3),
                nn.ReLU(),
                nn.MaxPool1d(4),
                
                nn.Conv1d(32, self.wave_embedding_size, kernel_size=5, stride=2),
                nn.ReLU(),
                
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            )
            
            self.wave_gru = nn.GRU(
                input_size=self.wave_embedding_size, 
                hidden_size=self.wave_embedding_size, 
                batch_first=True
            )
            
            fc_input_dim = self.cat_hidden_size + self.wave_embedding_size
        else:
            fc_input_dim = self.cat_hidden_size
            
        # Fusion & Classification
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1) 
        )

    def forward(self, cat_seq, wave_seq=None):
        """
        Args:
            cat_seq: Tensor of shape (batch, seq_len, 3)
            wave_seq: Tensor of shape (batch, seq_len, 3, hour_samples). Can be None if use_waveform=False.
        """
        batch_size, seq_len, _ = cat_seq.shape
        
        # Process Catalog
        cat_out, _ = self.catalog_gru(cat_seq)
        cat_final = cat_out[:, -1, :] 
        
        if not self.use_waveform:
            return self.classifier(cat_final)
            
        if wave_seq is None:
            raise ValueError("Model initialized with use_waveform=True, but wave_seq is None.")
            
        _, _, channels, samples = wave_seq.shape
        wave_flat = wave_seq.view(batch_size * seq_len, channels, samples)
        
        cnn_embeds = self.cnn_extractor(wave_flat)
        
        wave_embed_seq = cnn_embeds.view(batch_size, seq_len, self.wave_embedding_size)
        
        wave_out, _ = self.wave_gru(wave_embed_seq)
        wave_final = wave_out[:, -1, :]
        
        fused = torch.cat([cat_final, wave_final], dim=1)
        
        return self.classifier(fused)
