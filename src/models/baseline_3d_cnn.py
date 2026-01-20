"""
Baseline 3D CNN for comparison with recurrent models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline3DCNN(nn.Module):
    """
    Baseline 3D CNN that processes sequential scans as independent volumes.
    """
    
    def __init__(self, in_channels=4, base_features=32, output_channels=1):
        super(Baseline3DCNN, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv3d(in_channels, base_features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(base_features)
        
        self.conv2 = nn.Conv3d(base_features, base_features * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(base_features * 2)
        
        self.conv3 = nn.Conv3d(base_features * 2, base_features * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(base_features * 4)
        
        self.conv4 = nn.Conv3d(base_features * 4, base_features * 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(base_features * 8)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose3d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.dec1 = nn.Conv3d(base_features * 8, base_features * 4, kernel_size=3, padding=1)
        self.bn_dec1 = nn.BatchNorm3d(base_features * 4)
        
        self.upconv2 = nn.ConvTranspose3d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Conv3d(base_features * 4, base_features * 2, kernel_size=3, padding=1)
        self.bn_dec2 = nn.BatchNorm3d(base_features * 2)
        
        self.upconv3 = nn.ConvTranspose3d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec3 = nn.Conv3d(base_features * 2, base_features, kernel_size=3, padding=1)
        self.bn_dec3 = nn.BatchNorm3d(base_features)
        
        # Output layer
        self.out = nn.Conv3d(base_features, output_channels, kernel_size=1)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout3d(0.3)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, channels, depth, height, width)
        
        Returns:
            output: Predicted tumor state
        """
        # Encoder
        e1 = F.relu(self.bn1(self.conv1(x)))
        
        e2 = self.pool(e1)
        e2 = F.relu(self.bn2(self.conv2(e2)))
        
        e3 = self.pool(e2)
        e3 = F.relu(self.bn3(self.conv3(e3)))
        
        e4 = self.pool(e3)
        e4 = F.relu(self.bn4(self.conv4(e4)))
        e4 = self.dropout(e4)
        
        # Decoder with skip connections
        d1 = self.upconv1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = F.relu(self.bn_dec1(self.dec1(d1)))
        
        d2 = self.upconv2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = F.relu(self.bn_dec2(self.dec2(d2)))
        
        d3 = self.upconv3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = F.relu(self.bn_dec3(self.dec3(d3)))
        
        # Output
        output = self.out(d3)
        
        return output
