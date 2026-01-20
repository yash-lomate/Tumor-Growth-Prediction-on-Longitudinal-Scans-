"""
Main training script for tumor growth prediction models.
"""

import os
import argparse
import torch
import torch.nn as nn
from pathlib import Path

from src.models import Conv3DLSTM, Recurrent3DCNN, Baseline3DCNN
from src.data import LongitudinalDataLoader
from src.training import Trainer, CombinedLoss
from src.utils import Config, save_config


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


def main(args):
    """Main training function."""
    # Load configuration
    if args.config:
        config = Config.from_file(args.config)
    else:
        config = Config()
    
    # Override config with command line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.model_type:
        config.model_type = args.model_type
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save configuration
    config_save_path = os.path.join(config.output_dir, 'config.yaml')
    save_config(config, config_save_path)
    print(f"Configuration saved to {config_save_path}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = LongitudinalDataLoader.create_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_time_steps=config.num_time_steps,
        train_split=config.train_split,
        val_split=config.val_split,
        num_workers=config.num_workers
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"Creating model: {config.model_type}")
    model = get_model(config)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create loss function
    criterion = CombinedLoss(
        mse_weight=config.mse_weight,
        dice_weight=config.dice_weight,
        smooth_weight=config.smooth_weight
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        learning_rate=config.learning_rate,
        num_epochs=config.num_epochs,
        checkpoint_dir=config.checkpoint_dir,
        log_dir=config.log_dir
    )
    
    # Load checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    trainer.train()
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train tumor growth prediction model')
    
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, help='Path to data directory')
    parser.add_argument('--model_type', type=str, choices=['conv3d_lstm', 'recurrent_3d_cnn', 'baseline_3d_cnn'],
                        help='Type of model to train')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args)
