# Face Parsing

A face parsing model that combines VGG19 CNN features with DINOv3 visual representations for accurate facial segmentation.


<p align="center">
  <img src="./assets/21_visualization.png" width="1200" height="400"/>
  <img src="./assets/51_visualization.png" width="1200" height="400"/>
</p>

This example is the output of 11 face labels model. 

## Model Architecture

This model uses a dual-encoder architecture:
- **CNN Encoder**: VGG19 backbone for multi-scale feature extraction
- **Vision Transformer**: DINOv3 (ViT-L/16) for semantic feature extraction
- **Decoder**: Custom decoder that fuses features from both encoders

## Weights

### DINOv3 Weights
The DINOv3 weights (`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`) are from the official DINOv3 repository and should be downloaded separately:
- Source: [Official DINOv3 Repository](https://github.com/facebookresearch/dinov3)
- Model: ViT-L/16 pretrained on LVD-1689M dataset
- Place the weights in: `checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`

### Decoder Weights
Only the [decoder weights](https://drive.google.com/drive/folders/1qvMJ418YFywkksFhPFsjldYpgBwOO8RH?usp=drive_link) are provided in this repository (`experiments/checkpoints/decoder.ckpt`). These weights include:
- Custom decoder layers
- VGG19 CNN encoder (pretrained and fine-tuned)
- All training-specific parameters

The DINOv3 encoder is loaded separately to keep the checkpoint size manageable and respect the original model distribution.

## Face Parsing Labels

This model supports two label schemes depending on the `num_classes` argument:

### 19-Class Label Scheme (Default)
The standard **19-class face parsing scheme** provides detailed segmentation of facial components:

| Label | Class | Description |
|-------|-------|-------------|
| 0 | Background | Non-face regions |
| 1 | Skin | Face skin |
| 2 | Nose | Nose region |
| 3 | Right Eye | Right eye |
| 4 | Left Eye | Left eye |
| 5 | Right Brow | Right eyebrow |
| 6 | Left Brow | Left eyebrow |
| 7 | Right Ear | Right ear |
| 8 | Left Ear | Left ear |
| 9 | Mouth Interior | Inside of the mouth |
| 10 | Top Lip | Upper lip region |
| 11 | Bottom Lip | Lower lip region |
| 12 | Neck | Neck region |
| 13 | Hair | Hair region |
| 14 | Beard | Beard region |
| 15 | Clothing | Clothing |
| 16 | Glasses | Eyeglasses |
| 17 | Headwear | Hat, cap, etc. |
| 18 | Facewear | Face accessories |

### 11-Class Label Scheme (Simplified)
The **11-class OurFaceLabel** scheme provides a simplified but effective segmentation by merging similar regions:

| Label | Class | Description |
|-------|-------|-------------|
| 0 | Background | Non-face regions |
| 1 | Skin | Face, neck, and nose skin |
| 2 | Eyes | Both left and right eyes |
| 3 | Eyebrows | Both left and right eyebrows |
| 4 | Ears | Both left and right ears |
| 5 | Mouth Interior | Inside of the mouth |
| 6 | Upper Lip | Upper lip region |
| 7 | Lower Lip | Lower lip region |
| 8 | Hair | Hair region |
| 9 | Beard | Beard region |
| 10 | Accessory | Glasses, headwear, and face accessories |

The 11-class scheme simplifies the original 19 labels by:
- Merging left/right components (eyes, eyebrows, ears)
- Combining nose and neck into skin
- Grouping all accessories together
- Removing clothing from the segmentation

## Project Structure

```
FaceParsing/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── face_parsing_model.py      # PyTorch Lightning model
│   │   ├── encoder.py                 # VGG19 + DINOv3 encoder
│   │   └── decoder.py                 # Feature fusion decoder
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
│       └── decoder.ckpt               # Decoder-only checkpoint
├── checkpoints/
│   └── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth  # DINOv3 weights (download separately)
├── data/
│   ├── raw/                           # Raw dataset files
│   └── processed/                     # Processed data
├── train.py                           # Training script
├── inference.py                       # Inference script
├── extract_decoder_weights.py        # Script to extract decoder weights
├── requirements.txt                   # Dependencies
└── README.md
```

## Installation

```bash
# Install dependencies
pip install torch torchvision pytorch-lightning pillow numpy
```

## Usage

### Inference

Run inference with **19-class segmentation** (default):
```bash
python inference.py --input_dir path/to/image.jpg \
                   --output_dir outputs/ \
                   --checkpoint experiments/checkpoints/decoder19.ckpt \
                   --dinov3_checkpoint checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
                   --num_classes 19
```

Run inference with **11-class simplified segmentation**:
```bash
python inference.py --input_dir path/to/image.jpg \
                   --output_dir outputs/ \
                   --checkpoint experiments/checkpoints/decoder.ckpt \
                   --dinov3_checkpoint checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
                   --num_classes 11 \
                   --save_overlay
```

Run inference on a directory of images:
```bash
python inference.py --input_dir path/to/images/ \
                   --output_dir outputs/ \
                   --num_classes 19 \
                   --save_overlay
```


### Extract Decoder Weights (if using full checkpoint)
If you have a full checkpoint (`last.ckpt`) that includes DINOv3 weights, extract the decoder-only weights:

```bash
python extract_decoder_weights.py --checkpoint experiments/checkpoints/last.ckpt \
                                  --output experiments/checkpoints/decoder.ckpt
```

### Command Line Arguments

- `--input_dir`: Path to input image or directory
- `--output_dir`: Path to output directory (default: `./outputs`)
- `--checkpoint`: Path to decoder checkpoint (default: `experiments/checkpoints/decoder.ckpt`)
- `--dinov3_checkpoint`: Path to DINOv3 checkpoint (default: `checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`)
- `--device`: Device to run on (`cuda` or `cpu`)
- `--image_size`: Input image size as height width (default: 512 512)
- `--num_classes`: Number of segmentation classes (default: 19)
  - Use `19` for detailed 19-class segmentation
  - Use `11` for simplified OurFaceLabel segmentation
- `--save_overlay`: Save overlay visualization
- `--alpha`: Transparency for overlay visualization (default: 0.5)

## Output

The model generates:
- `*_mask.png`: Color-coded segmentation mask overlaid on the original image
- `*_visualization.png`: Side-by-side comparison of input and segmentation (if `--save_overlay` is used)

## Training

For training details, refer to the training scripts in the repository. The model uses:
- PyTorch Lightning for training orchestration
- Cross-entropy loss with optional auxiliary losses
- AdamW optimizer with cosine annealing scheduler

### Training Command

Train with 19-class segmentation (default):
```bash
python train.py \
    --data_dir ./data \
    --batch_size 8 \
    --max_epochs 100 \
    --num_classes 19 \
    --gpus 1
```

Train with 11-class simplified segmentation:
```bash
python train.py \
    --data_dir ./data \
    --batch_size 8 \
    --max_epochs 100 \
    --num_classes 11 \
    --gpus 1
```

## Features

- **Modular Architecture**: Clean separation between model, data, and training logic
- **Dual Encoder Design**: Combines CNN and Vision Transformer features
- **PyTorch Lightning**: Leverages Lightning for training loop abstraction
- **Configuration Management**: YAML-based configuration system
- **Visualization Tools**: Built-in visualization for predictions and results
- **Flexible Data Pipeline**: Easily extensible data module with augmentation support
- **Inference Pipeline**: Ready-to-use inference script for single images or directories

## Citation

If you use this model, please cite:
- Original DINOv3 paper for the vision transformer encoder
- This repository for the face parsing implementation

## License
DINOv3 weights are subject to the original Meta AI license terms.