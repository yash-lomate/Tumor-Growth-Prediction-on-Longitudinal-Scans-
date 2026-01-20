"""
Visualization utilities for tumor growth prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path


def visualize_prediction(input_scans, prediction, target, save_path=None, slice_idx=None):
    """
    Visualize input scans, prediction, and target.
    
    Args:
        input_scans: Input scans tensor (time_steps, channels, D, H, W)
        prediction: Model prediction (channels, D, H, W)
        target: Ground truth (channels, D, H, W)
        save_path: Path to save visualization
        slice_idx: Slice index to visualize (default: middle slice)
    """
    # Convert to numpy
    if isinstance(input_scans, torch.Tensor):
        input_scans = input_scans.detach().cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Get middle slice if not specified
    if slice_idx is None:
        slice_idx = input_scans.shape[2] // 2
    
    # Extract slices
    num_time_steps = input_scans.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(3, num_time_steps + 2, figsize=(4 * (num_time_steps + 2), 12))
    
    # Plot input scans at each time step
    for t in range(num_time_steps):
        # Plot axial slice
        axes[0, t].imshow(input_scans[t, 0, slice_idx, :, :], cmap='gray')
        axes[0, t].set_title(f'Input T{t}')
        axes[0, t].axis('off')
        
        # Plot coronal slice
        axes[1, t].imshow(input_scans[t, 0, :, slice_idx, :], cmap='gray')
        axes[1, t].axis('off')
        
        # Plot sagittal slice
        axes[2, t].imshow(input_scans[t, 0, :, :, slice_idx], cmap='gray')
        axes[2, t].axis('off')
    
    # Plot prediction
    axes[0, num_time_steps].imshow(prediction[0, slice_idx, :, :], cmap='hot')
    axes[0, num_time_steps].set_title('Prediction')
    axes[0, num_time_steps].axis('off')
    
    axes[1, num_time_steps].imshow(prediction[0, :, slice_idx, :], cmap='hot')
    axes[1, num_time_steps].axis('off')
    
    axes[2, num_time_steps].imshow(prediction[0, :, :, slice_idx], cmap='hot')
    axes[2, num_time_steps].axis('off')
    
    # Plot target
    axes[0, num_time_steps + 1].imshow(target[0, slice_idx, :, :], cmap='hot')
    axes[0, num_time_steps + 1].set_title('Ground Truth')
    axes[0, num_time_steps + 1].axis('off')
    
    axes[1, num_time_steps + 1].imshow(target[0, :, slice_idx, :], cmap='hot')
    axes[1, num_time_steps + 1].axis('off')
    
    axes[2, num_time_steps + 1].imshow(target[0, :, :, slice_idx], cmap='hot')
    axes[2, num_time_steps + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Losses', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def save_prediction_slices(prediction, save_dir, prefix='prediction'):
    """
    Save multiple slices of a prediction volume.
    
    Args:
        prediction: Prediction volume (C, D, H, W)
        save_dir: Directory to save slices
        prefix: Prefix for saved files
    """
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get dimensions
    depth = prediction.shape[1]
    
    # Save slices at different depths
    slice_indices = [depth // 4, depth // 2, 3 * depth // 4]
    
    for idx in slice_indices:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Axial view
        axes[0].imshow(prediction[0, idx, :, :], cmap='hot')
        axes[0].set_title(f'Axial (slice {idx})')
        axes[0].axis('off')
        
        # Coronal view
        axes[1].imshow(prediction[0, :, idx, :], cmap='hot')
        axes[1].set_title(f'Coronal (slice {idx})')
        axes[1].axis('off')
        
        # Sagittal view
        axes[2].imshow(prediction[0, :, :, idx], cmap='hot')
        axes[2].set_title(f'Sagittal (slice {idx})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{prefix}_slice_{idx}.png', dpi=150, bbox_inches='tight')
        plt.close()


def create_gif_from_predictions(predictions, save_path, duration=200):
    """
    Create GIF from sequence of predictions.
    
    Args:
        predictions: List of prediction volumes
        save_path: Path to save GIF
        duration: Duration per frame in milliseconds
    """
    from PIL import Image
    
    frames = []
    
    for pred in predictions:
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        
        # Get middle slice
        slice_idx = pred.shape[1] // 2
        slice_img = pred[0, slice_idx, :, :]
        
        # Normalize to 0-255
        slice_img = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8) * 255).astype(np.uint8)
        
        # Convert to PIL Image
        img = Image.fromarray(slice_img)
        frames.append(img)
    
    # Save as GIF
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
