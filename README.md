# Face Parsing with PyTorch Lightning

A modular face parsing implementation using PyTorch Lightning framework.

## Project Structure

```
FaceParsing/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── face_parsing_model.py      # Base model implementation
│   ├── data/
│   │   ├── __init__.py
│   │   └── face_parsing_datamodule.py # Data loading and preprocessing
│   ├── configs/
│   │   └── default_config.yaml        # Default configuration
│   └── utils/
│       ├── __init__.py
│       ├── config_loader.py           # Configuration management
│       └── visualization.py           # Visualization utilities
├── experiments/
│   ├── logs/                          # TensorBoard logs
│   └── checkpoints/                   # Model checkpoints
├── data/
│   ├── raw/                           # Raw dataset files
│   └── processed/                     # Processed data
├── train.py                           # Training script
├── inference.py                       # Inference script
├── requirements.txt                   # Dependencies
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Features

- **Modular Architecture**: Clean separation between model, data, and training logic
- **PyTorch Lightning**: Leverages Lightning for training loop abstraction
- **Configuration Management**: YAML-based configuration system
- **Visualization Tools**: Built-in visualization for predictions and results
- **Flexible Data Pipeline**: Easily extensible data module with augmentation support
- **Inference Pipeline**: Ready-to-use inference script for single images or directories

## Usage

### Training

```bash
python train.py \
    --data_dir ./data \
    --batch_size 8 \
    --max_epochs 100 \
    --gpus 1
```

### Inference

```bash
# Single image
python inference.py \
    --input path/to/image.jpg \
    --checkpoint path/to/checkpoint.ckpt \
    --output ./outputs \
    --save_overlay

# Directory of images
python inference.py \
    --input path/to/image_dir/ \
    --checkpoint path/to/checkpoint.ckpt \
    --output ./outputs
```

## Configuration

The default configuration is stored in `src/configs/default_config.yaml`. Key parameters include:

- **Data**: Image size, batch size, augmentation settings
- **Model**: Number of classes, backbone architecture (to be implemented)
- **Training**: Learning rate, optimizer, scheduler settings
- **Hardware**: GPU configuration, mixed precision training

## Next Steps

1. **Implement Network Architecture**: Replace placeholder encoder/decoder in `face_parsing_model.py`
2. **Dataset Integration**: Implement actual data loading in `face_parsing_datamodule.py`
3. **Add Model Architectures**: Implement specific architectures (U-Net, DeepLab, BiSeNet, etc.)
4. **Metrics Enhancement**: Add more evaluation metrics (per-class IoU, F1-score)
5. **Data Augmentation**: Extend augmentation pipeline for better generalization

## Dataset Support

Currently configured for CelebAMask-HQ dataset with 19 classes:
- Background, skin, eyebrows, eyes, ears, nose, mouth, lips, neck, cloth, hair, hat

## License

[Add your license here]