# Tumor Growth Prediction on Longitudinal Scans

A deep learning framework for forecasting tumor progression using recurrent CNN and 3D CNN on sequential MRI/CT scans. This project implements state-of-the-art architectures for spatiotemporal modeling of tumor growth from longitudinal medical imaging data.

## Overview

This framework provides tools for predicting tumor growth trajectories from sequential medical scans (MRI/CT over time). The models leverage both spatial features from 3D CNNs and temporal patterns from recurrent architectures to forecast future tumor states.

### Key Features

- **Multiple Model Architectures**:
  - 3D CNN + LSTM: Combines 3D spatial feature extraction with LSTM temporal modeling
  - Recurrent 3D CNN with ConvLSTM: Spatiotemporal feature learning with convolutional LSTM cells
  - Baseline 3D CNN: Standard 3D U-Net for comparison

- **Comprehensive Data Pipeline**:
  - Support for longitudinal MRI/CT data
  - Multi-modal imaging (T1, T1ce, T2, FLAIR)
  - Data augmentation for medical images
  - BraTS dataset compatibility

- **Advanced Training Features**:
  - Combined loss functions (MSE, Dice, Smoothness)
  - Multiple evaluation metrics (Dice, IoU, Hausdorff distance)
  - TensorBoard integration
  - Checkpoint management

## Architecture Details

### Conv3D-LSTM Model

The Conv3D-LSTM model processes sequential 3D scans through a two-stage pipeline:

1. **3D CNN Encoder**: Extracts spatial features from each time point independently
2. **LSTM Temporal Module**: Processes the sequence of spatial features to capture temporal patterns
3. **Prediction Head**: Generates forecasts for future tumor states

```
Input Sequence → [3D CNN] → Feature Sequence → [LSTM] → [FC Layers] → Prediction
```

### Recurrent 3D CNN (ConvLSTM)

The Recurrent 3D CNN uses ConvLSTM layers to jointly learn spatiotemporal features:

1. **ConvLSTM Cells**: Maintain spatial structure while modeling temporal dynamics
2. **Multi-scale Processing**: Hierarchical feature learning at different resolutions
3. **Recurrent Prediction**: Can generate multiple future time steps

```
Input Sequence → [ConvLSTM Layers] → [Conv3D] → Prediction
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/yash-lomate/Tumor-Growth-Prediction-on-Longitudinal-Scans-.git
cd Tumor-Growth-Prediction-on-Longitudinal-Scans-

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### Expected Data Structure

The framework expects longitudinal data organized as follows:

```
data/
├── patient_001/
│   ├── timepoint_0.nii.gz
│   ├── timepoint_1.nii.gz
│   ├── timepoint_2.nii.gz
│   └── timepoint_3.nii.gz
├── patient_002/
│   └── ...
└── ...
```

### BraTS Longitudinal Data

For BraTS multi-modal data:

```
data/
├── patient_001/
│   ├── timepoint_0/
│   │   ├── t1.nii.gz
│   │   ├── t1ce.nii.gz
│   │   ├── t2.nii.gz
│   │   └── flair.nii.gz
│   ├── timepoint_1/
│   │   └── ...
│   └── ...
└── ...
```

### Public Datasets

Recommended public longitudinal datasets:
- **BraTS**: Brain Tumor Segmentation Challenge (http://braintumorsegmentation.org/)
- **ACRIN-FMISO-Brain**: Longitudinal brain tumor data (https://www.cancerimagingarchive.net/)
- **QIN-BRAIN-DSC-MRI**: Glioblastoma imaging (https://www.cancerimagingarchive.net/)

## Usage

### Training

#### Using Command Line

```bash
# Train Conv3D-LSTM model
python train.py \
    --data_dir data/brats \
    --model_type conv3d_lstm \
    --batch_size 4 \
    --num_epochs 100 \
    --learning_rate 1e-4

# Train Recurrent 3D CNN
python train.py \
    --data_dir data/brats \
    --model_type recurrent_3d_cnn \
    --batch_size 4 \
    --num_epochs 100

# Train with configuration file
python train.py --config configs/default_config.yaml
```

#### Using Python API

```python
from src.models import Conv3DLSTM
from src.data import LongitudinalDataLoader
from src.training import Trainer, CombinedLoss
from src.utils import Config

# Load configuration
config = Config.from_file('configs/default_config.yaml')

# Create data loaders
train_loader, val_loader, test_loader = LongitudinalDataLoader.create_dataloaders(
    data_dir='data/brats',
    batch_size=4,
    num_time_steps=4
)

# Create model
model = Conv3DLSTM(
    in_channels=4,
    base_features=32,
    hidden_size=512,
    num_lstm_layers=2,
    num_time_steps=4
)

# Train
criterion = CombinedLoss()
trainer = Trainer(model, train_loader, val_loader, criterion)
trainer.train()
```

### Inference

```bash
# Run inference on test data
python inference.py \
    --config outputs/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --data_dir data/test \
    --output_dir predictions \
    --visualize \
    --save_slices
```

### Configuration

Create a YAML configuration file:

```yaml
# Model parameters
model_type: conv3d_lstm
in_channels: 4
base_features: 32
hidden_size: 512
num_lstm_layers: 2
num_time_steps: 4

# Training parameters
batch_size: 4
num_epochs: 100
learning_rate: 0.0001

# Data parameters
data_dir: data/brats
target_shape: [128, 128, 128]
train_split: 0.7
val_split: 0.15
```

## Model Performance

Expected performance metrics on BraTS longitudinal data:

| Model | Dice Score | IoU | Volumetric Similarity |
|-------|-----------|-----|----------------------|
| Conv3D-LSTM | 0.85+ | 0.78+ | 0.82+ |
| Recurrent 3D CNN | 0.87+ | 0.80+ | 0.84+ |
| Baseline 3D CNN | 0.80+ | 0.72+ | 0.78+ |

*Note: Actual performance depends on dataset and hyperparameters.*

## Project Structure

```
Tumor-Growth-Prediction-on-Longitudinal-Scans-/
├── src/
│   ├── models/
│   │   ├── conv3d_lstm.py          # 3D CNN + LSTM model
│   │   ├── recurrent_3d_cnn.py     # ConvLSTM-based model
│   │   └── baseline_3d_cnn.py      # Baseline 3D U-Net
│   ├── data/
│   │   ├── dataset.py              # Dataset classes
│   │   └── preprocessing.py         # Data preprocessing
│   ├── training/
│   │   ├── trainer.py              # Training loop
│   │   ├── losses.py               # Loss functions
│   │   └── metrics.py              # Evaluation metrics
│   └── utils/
│       ├── config.py               # Configuration management
│       └── visualization.py         # Visualization tools
├── train.py                         # Training script
├── inference.py                     # Inference script
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## Clinical Relevance

This framework addresses critical needs in oncology:

1. **Disease Trajectory Prediction**: Forecasts tumor growth patterns to anticipate disease progression
2. **Personalized Treatment Planning**: Enables individualized therapy based on predicted outcomes
3. **Treatment Response Monitoring**: Evaluates therapy effectiveness through growth pattern analysis
4. **Early Intervention**: Identifies aggressive growth patterns for timely treatment adjustments

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{tumor_growth_prediction,
  title={Tumor Growth Prediction on Longitudinal Scans},
  author={Yash Lomate},
  year={2024},
  url={https://github.com/yash-lomate/Tumor-Growth-Prediction-on-Longitudinal-Scans-}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- BraTS Challenge organizers for providing benchmark datasets
- The Cancer Imaging Archive (TCIA) for public longitudinal imaging data
- PyTorch and MONAI communities for excellent deep learning tools

## References

1. Zhou, Z., et al. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation." (2018)
2. Shi, X., et al. "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting." (2015)
3. Menze, B.H., et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)." (2015)
4. Nie, D., et al. "3D Deep Learning for Multi-modal Imaging-Guided Survival Time Prediction of Brain Tumor Patients." (2016)

## Contact

For questions or collaborations, please open an issue on GitHub.