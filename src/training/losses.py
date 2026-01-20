"""
Loss functions for tumor growth prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.
    """
    
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model predictions (B, C, D, H, W)
            targets: Ground truth (B, C, D, H, W)
        """
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model predictions (B, C, D, H, W)
            targets: Ground truth (B, C, D, H, W)
        """
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function for tumor growth prediction.
    Combines MSE, Dice, and perceptual losses.
    """
    
    def __init__(self, mse_weight=1.0, dice_weight=1.0, smooth_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.dice_weight = dice_weight
        self.smooth_weight = smooth_weight
        
        self.mse = nn.MSELoss()
        self.dice = DiceLoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model predictions (B, C, D, H, W)
            targets: Ground truth (B, C, D, H, W)
        """
        # MSE loss for intensity matching
        mse_loss = self.mse(predictions, targets)
        
        # Dice loss for structural similarity
        dice_loss = self.dice(torch.sigmoid(predictions), targets)
        
        # Smoothness loss to encourage spatially coherent predictions
        smooth_loss = self._total_variation_loss(predictions)
        
        # Combined loss
        total_loss = (
            self.mse_weight * mse_loss +
            self.dice_weight * dice_loss +
            self.smooth_weight * smooth_loss
        )
        
        return total_loss
    
    def _total_variation_loss(self, x):
        """Compute total variation loss for smoothness."""
        batch_size = x.size(0)
        
        # Compute differences in each dimension
        diff_i = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        diff_j = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        diff_k = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
        
        tv_loss = (diff_i.sum() + diff_j.sum() + diff_k.sum()) / batch_size
        return tv_loss


class TemporalConsistencyLoss(nn.Module):
    """
    Loss to enforce temporal consistency in predictions.
    """
    
    def __init__(self, weight=0.1):
        super(TemporalConsistencyLoss, self).__init__()
        self.weight = weight
    
    def forward(self, predictions_sequence):
        """
        Args:
            predictions_sequence: Sequence of predictions (B, T, C, D, H, W)
        """
        if predictions_sequence.size(1) < 2:
            return torch.tensor(0.0, device=predictions_sequence.device)
        
        # Compute temporal differences
        temporal_diff = predictions_sequence[:, 1:] - predictions_sequence[:, :-1]
        
        # Penalize large temporal changes
        consistency_loss = torch.mean(torch.abs(temporal_diff))
        
        return self.weight * consistency_loss
