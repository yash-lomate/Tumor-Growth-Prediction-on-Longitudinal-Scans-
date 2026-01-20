"""
Inference script for tumor growth prediction.
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path

from src.models import Conv3DLSTM, Recurrent3DCNN, Baseline3DCNN
from src.data import TumorGrowthDataset
from src.training import compute_metrics
from src.utils import Config, visualize_prediction, save_prediction_slices


def get_model(config):
    """Create model based on configuration."""
    if config.model_type == 'conv3d_lstm':
        model = Conv3DLSTM(
            in_channels=config.in_channels,
            base_features=config.base_features,
            hidden_size=config.hidden_size,
            num_lstm_layers=config.num_lstm_layers,
            num_time_steps=config.num_time_steps,
            output_channels=config.output_channels
        )
    elif config.model_type == 'recurrent_3d_cnn':
        model = Recurrent3DCNN(
            in_channels=config.in_channels,
            hidden_channels=[config.base_features, config.base_features * 2, config.base_features * 4],
            num_layers=3,
            output_channels=config.output_channels
        )
    elif config.model_type == 'baseline_3d_cnn':
        model = Baseline3DCNN(
            in_channels=config.in_channels,
            base_features=config.base_features,
            output_channels=config.output_channels
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    return model


def predict(model, inputs, device):
    """Make prediction."""
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        prediction = model(inputs)
    return prediction


def main(args):
    """Main inference function."""
    # Load configuration
    config = Config.from_file(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print(f"Creating model: {config.model_type}")
    model = get_model(config)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    print("Loading test data...")
    test_dataset = TumorGrowthDataset(
        data_dir=args.data_dir,
        num_time_steps=config.num_time_steps,
        target_shape=tuple(config.target_shape)
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Run inference on all test samples
    all_metrics = []
    
    for i in range(len(test_dataset)):
        print(f"\nProcessing sample {i + 1}/{len(test_dataset)}...")
        
        # Get sample
        inputs, target = test_dataset[i]
        inputs = inputs.unsqueeze(0)  # Add batch dimension
        target = target.unsqueeze(0)
        
        # Make prediction
        prediction = predict(model, inputs, device)
        
        # Compute metrics
        metrics = compute_metrics(prediction, target.to(device))
        all_metrics.append(metrics)
        
        print(f"Dice: {metrics['dice']:.4f}")
        print(f"IoU: {metrics['iou']:.4f}")
        print(f"Volumetric Similarity: {metrics['volumetric_similarity']:.4f}")
        
        # Save visualization
        if args.visualize:
            vis_path = output_dir / f'prediction_{i}.png'
            visualize_prediction(
                inputs[0], prediction[0], target[0],
                save_path=str(vis_path)
            )
            print(f"Visualization saved to {vis_path}")
        
        # Save prediction slices
        if args.save_slices:
            slice_dir = output_dir / f'slices_{i}'
            save_prediction_slices(prediction[0], slice_dir, prefix=f'pred_{i}')
            print(f"Slices saved to {slice_dir}")
    
    # Compute average metrics
    print("\n" + "=" * 50)
    print("Average Metrics:")
    print("=" * 50)
    
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if m[key] != float('inf')]
        if values:
            avg_metrics[key] = np.mean(values)
            std_metrics = np.std(values)
            print(f"{key}: {avg_metrics[key]:.4f} ± {std_metrics:.4f}")
    
    # Save metrics
    metrics_path = output_dir / 'metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("Average Metrics:\n")
        f.write("=" * 50 + "\n")
        for key, value in avg_metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    print(f"\nMetrics saved to {metrics_path}")
    print("Inference completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with tumor growth prediction model')
    
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Save visualizations')
    parser.add_argument('--save_slices', action='store_true', help='Save prediction slices')
    
    args = parser.parse_args()
    main(args)
