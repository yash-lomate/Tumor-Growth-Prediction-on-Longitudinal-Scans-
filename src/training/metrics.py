"""
Evaluation metrics for tumor growth prediction.
"""

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff


def dice_coefficient(predictions, targets, threshold=0.5):
    """
    Compute Dice coefficient.
    
    Args:
        predictions: Model predictions
        targets: Ground truth
        threshold: Threshold for binarization
    
    Returns:
        Dice coefficient
    """
    # Binarize predictions
    predictions = (predictions > threshold).float()
    targets = (targets > threshold).float()
    
    # Flatten
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    intersection = (predictions * targets).sum()
    dice = (2. * intersection) / (predictions.sum() + targets.sum() + 1e-8)
    
    return dice.item()


def iou_score(predictions, targets, threshold=0.5):
    """
    Compute Intersection over Union (IoU) score.
    
    Args:
        predictions: Model predictions
        targets: Ground truth
        threshold: Threshold for binarization
    
    Returns:
        IoU score
    """
    predictions = (predictions > threshold).float()
    targets = (targets > threshold).float()
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    iou = intersection / (union + 1e-8)
    return iou.item()


def hausdorff_distance(predictions, targets, threshold=0.5):
    """
    Compute Hausdorff distance between predicted and target segmentations.
    
    Args:
        predictions: Model predictions (numpy array)
        targets: Ground truth (numpy array)
        threshold: Threshold for binarization
    
    Returns:
        Hausdorff distance
    """
    predictions = (predictions > threshold).astype(np.float32)
    targets = (targets > threshold).astype(np.float32)
    
    # Get surface points
    pred_points = np.argwhere(predictions)
    target_points = np.argwhere(targets)
    
    if len(pred_points) == 0 or len(target_points) == 0:
        return float('inf')
    
    # Compute Hausdorff distance
    hd1 = directed_hausdorff(pred_points, target_points)[0]
    hd2 = directed_hausdorff(target_points, pred_points)[0]
    
    return max(hd1, hd2)


def volumetric_similarity(predictions, targets, threshold=0.5):
    """
    Compute volumetric similarity.
    
    Args:
        predictions: Model predictions
        targets: Ground truth
        threshold: Threshold for binarization
    
    Returns:
        Volumetric similarity score
    """
    predictions = (predictions > threshold).float()
    targets = (targets > threshold).float()
    
    pred_volume = predictions.sum()
    target_volume = targets.sum()
    
    vs = 1 - abs(pred_volume - target_volume) / (pred_volume + target_volume + 1e-8)
    return vs.item()


def compute_metrics(predictions, targets, threshold=0.5):
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth
        threshold: Threshold for binarization
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'dice': dice_coefficient(predictions, targets, threshold),
        'iou': iou_score(predictions, targets, threshold),
        'volumetric_similarity': volumetric_similarity(predictions, targets, threshold)
    }
    
    # Compute Hausdorff distance (requires numpy)
    if isinstance(predictions, torch.Tensor):
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
    else:
        predictions_np = predictions
        targets_np = targets
    
    try:
        metrics['hausdorff'] = hausdorff_distance(predictions_np, targets_np, threshold)
    except:
        metrics['hausdorff'] = float('inf')
    
    return metrics


def mean_squared_error(predictions, targets):
    """Compute Mean Squared Error."""
    mse = torch.mean((predictions - targets) ** 2)
    return mse.item()


def mean_absolute_error(predictions, targets):
    """Compute Mean Absolute Error."""
    mae = torch.mean(torch.abs(predictions - targets))
    return mae.item()
