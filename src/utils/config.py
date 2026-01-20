"""
Configuration management for tumor growth prediction.
"""

import yaml
from pathlib import Path


class Config:
    """Configuration class for model training and inference."""
    
    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = self.get_default_config()
        
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    @staticmethod
    def get_default_config():
        """Get default configuration."""
        return {
            # Model parameters
            'model_type': 'conv3d_lstm',  # 'conv3d_lstm', 'recurrent_3d_cnn', 'baseline_3d_cnn'
            'in_channels': 4,
            'base_features': 32,
            'hidden_size': 512,
            'num_lstm_layers': 2,
            'num_time_steps': 4,
            'output_channels': 1,
            
            # Data parameters
            'data_dir': 'data/brats',
            'target_shape': [128, 128, 128],
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            
            # Training parameters
            'batch_size': 4,
            'num_epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'num_workers': 4,
            
            # Loss parameters
            'mse_weight': 1.0,
            'dice_weight': 1.0,
            'smooth_weight': 0.1,
            
            # Augmentation parameters
            'use_augmentation': True,
            'rotation_range': 10,
            'flip_prob': 0.5,
            'noise_std': 0.01,
            'blur_sigma': 0.5,
            'brightness_range': 0.2,
            
            # Paths
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
            'output_dir': 'outputs',
            
            # Device
            'device': 'cuda',
            'mixed_precision': True,
            
            # Evaluation
            'eval_interval': 5,
            'save_predictions': True
        }
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, filepath):
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_file(cls, filepath):
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Config object
    """
    return Config.from_file(config_path)


def save_config(config, filepath):
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object or dictionary
        filepath: Path to save configuration
    """
    if isinstance(config, Config):
        config.save(filepath)
    else:
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
