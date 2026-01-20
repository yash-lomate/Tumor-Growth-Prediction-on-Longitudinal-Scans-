"""
3D CNN with LSTM for temporal modeling of tumor growth on longitudinal scans.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """3D Convolutional block with batch normalization and activation."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv3DEncoder(nn.Module):
    """3D CNN encoder to extract spatial features from 3D medical scans."""
    
    def __init__(self, in_channels=4, base_features=32):
        super(Conv3DEncoder, self).__init__()
        
        # Encoder path
        self.enc1 = Conv3DBlock(in_channels, base_features)
        self.enc2 = Conv3DBlock(base_features, base_features * 2)
        self.enc3 = Conv3DBlock(base_features * 2, base_features * 4)
        self.enc4 = Conv3DBlock(base_features * 4, base_features * 8)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Global average pooling for feature vector
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
    def forward(self, x):
        # x shape: (batch, channels, depth, height, width)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Global pooling to get feature vector
        features = self.global_pool(e4)
        features = features.view(features.size(0), -1)
        
        return features, (e1, e2, e3, e4)


class Conv3DLSTM(nn.Module):
    """
    3D CNN + LSTM for tumor growth prediction on longitudinal scans.
    
    Architecture:
    1. 3D CNN encoder extracts spatial features from each time point
    2. LSTM processes temporal sequence of features
    3. Decoder predicts future tumor state
    """
    
    def __init__(self, in_channels=4, base_features=32, hidden_size=512, 
                 num_lstm_layers=2, num_time_steps=4, output_channels=1):
        super(Conv3DLSTM, self).__init__()
        
        self.num_time_steps = num_time_steps
        self.hidden_size = hidden_size
        
        # 3D CNN encoder for spatial feature extraction
        self.encoder = Conv3DEncoder(in_channels, base_features)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=base_features * 8,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.3 if num_lstm_layers > 1 else 0
        )
        
        # Decoder: project LSTM output to spatial prediction
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        
        # Output projection
        self.output_proj = nn.Linear(64, output_channels)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, time_steps, channels, depth, height, width)
        
        Returns:
            prediction: Predicted tumor growth (batch, output_channels)
        """
        batch_size, time_steps, c, d, h, w = x.shape
        
        # Extract features from each time step using 3D CNN
        features_list = []
        for t in range(time_steps):
            features, _ = self.encoder(x[:, t, :, :, :, :])
            features_list.append(features.unsqueeze(1))
        
        # Stack features across time
        temporal_features = torch.cat(features_list, dim=1)  # (batch, time_steps, features)
        
        # Process temporal sequence with LSTM
        lstm_out, (hidden, cell) = self.lstm(temporal_features)
        
        # Use last time step output
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Decode to prediction
        out = F.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = F.relu(self.fc3(out))
        
        prediction = self.output_proj(out)
        
        return prediction
    
    def predict_sequence(self, x):
        """
        Predict tumor growth for multiple future time steps.
        
        Args:
            x: Input tensor of shape (batch, time_steps, channels, depth, height, width)
        
        Returns:
            predictions: List of predictions for future time steps
        """
        batch_size, time_steps, c, d, h, w = x.shape
        
        # Extract features from each time step
        features_list = []
        for t in range(time_steps):
            features, _ = self.encoder(x[:, t, :, :, :, :])
            features_list.append(features.unsqueeze(1))
        
        temporal_features = torch.cat(features_list, dim=1)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(temporal_features)
        
        # Predict for each future time step
        predictions = []
        for t in range(lstm_out.size(1)):
            out = F.relu(self.fc1(lstm_out[:, t, :]))
            out = F.relu(self.fc2(out))
            out = F.relu(self.fc3(out))
            pred = self.output_proj(out)
            predictions.append(pred)
        
        return torch.stack(predictions, dim=1)
