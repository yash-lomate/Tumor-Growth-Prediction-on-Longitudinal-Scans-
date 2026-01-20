# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Prepare Your Data

Organize your longitudinal scans in this structure:
```
data/
├── patient_001/
│   ├── timepoint_0.nii.gz
│   ├── timepoint_1.nii.gz
│   ├── timepoint_2.nii.gz
│   └── timepoint_3.nii.gz
└── patient_002/
    └── ...
```

### 2. Train a Model

```bash
# Train Conv3D-LSTM model (default)
python train.py --data_dir data/your_data --model_type conv3d_lstm

# Train Recurrent 3D CNN
python train.py --data_dir data/your_data --model_type recurrent_3d_cnn

# Train with custom configuration
python train.py --config configs/custom_config.yaml
```

### 3. Run Inference

```bash
python inference.py \
    --config outputs/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --data_dir data/test \
    --output_dir predictions \
    --visualize
```

## Model Options

### Conv3D-LSTM
- **Best for**: Sequential feature extraction + temporal modeling
- **Architecture**: 3D CNN encoder → LSTM → Prediction head
- **Memory**: Moderate
- **Speed**: Fast

### Recurrent 3D CNN (ConvLSTM)
- **Best for**: Joint spatiotemporal feature learning
- **Architecture**: ConvLSTM layers → Output projection
- **Memory**: High
- **Speed**: Moderate

### Baseline 3D CNN
- **Best for**: Single time-point prediction (baseline comparison)
- **Architecture**: 3D U-Net
- **Memory**: Low
- **Speed**: Very fast

## Configuration Parameters

Key parameters to adjust in `configs/default_config.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_type` | Model architecture | `conv3d_lstm` |
| `num_time_steps` | Sequential scans to use | `4` |
| `batch_size` | Training batch size | `4` |
| `learning_rate` | Optimizer learning rate | `1e-4` |
| `num_epochs` | Training epochs | `100` |
| `target_shape` | Volume dimensions | `[128, 128, 128]` |

## Example: Training on BraTS Data

```bash
# 1. Download BraTS longitudinal data
# 2. Organize data
python train.py \
    --data_dir data/brats_longitudinal \
    --model_type conv3d_lstm \
    --batch_size 2 \
    --num_epochs 50 \
    --learning_rate 1e-4

# 3. Monitor training
tensorboard --logdir logs/

# 4. Run inference
python inference.py \
    --config outputs/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --data_dir data/brats_test \
    --visualize
```

## Performance Tips

1. **GPU Memory**: Reduce `batch_size` or `target_shape` if running out of memory
2. **Training Speed**: Use mixed precision training with `mixed_precision: true`
3. **Data Augmentation**: Enable with `use_augmentation: true` for better generalization
4. **Multi-GPU**: Set `CUDA_VISIBLE_DEVICES` environment variable

## Troubleshooting

### Out of Memory (OOM)
```yaml
batch_size: 2  # Reduce batch size
target_shape: [96, 96, 96]  # Reduce volume size
```

### Slow Training
```yaml
num_workers: 8  # Increase data loading workers
mixed_precision: true  # Enable mixed precision
```

### Poor Performance
```yaml
use_augmentation: true  # Enable data augmentation
num_epochs: 200  # Train longer
learning_rate: 5e-5  # Try different learning rate
```

## Advanced Usage

### Custom Loss Function

```python
from src.training import Trainer, CombinedLoss

criterion = CombinedLoss(
    mse_weight=1.0,
    dice_weight=2.0,  # Increase dice weight
    smooth_weight=0.05
)
```

### Custom Data Augmentation

```python
from src.data.preprocessing import AugmentationTransform

augmentation_params = {
    'rotation_range': 15,
    'flip_prob': 0.6,
    'noise_std': 0.02
}
transform = AugmentationTransform(augmentation_params)
```

### Resume Training

```bash
python train.py \
    --config outputs/config.yaml \
    --resume checkpoints/checkpoint_epoch_50.pth
```

## Citation

```bibtex
@software{tumor_growth_prediction,
  title={Tumor Growth Prediction on Longitudinal Scans},
  author={Yash Lomate},
  year={2024},
  url={https://github.com/yash-lomate/Tumor-Growth-Prediction-on-Longitudinal-Scans-}
}
```
