import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import random
import torchvision.transforms.functional as TF


class FaceParsingDataset(Dataset):
    """
    Dataset class for Face Parsing task.
    This is a placeholder implementation that needs to be extended with actual data loading logic.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (512, 512),
        augmentation: bool = True
    ):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.augmentation = augmentation and (split == 'train')
        
        img_dir = os.path.join(data_dir, "images")
        seg_dir = os.path.join(data_dir, "segmentations")        

        self.img_paths = []
        self.seg_paths = []

        img_names = sorted(os.listdir(img_dir))
        if split != "train":
            img_names = img_names[::1000]
        for img_name in img_names:
            img_path = os.path.join(img_dir, img_name)
            seg_path = os.path.join(seg_dir, img_name.replace(".png", "_seg.png"))

            if os.path.exists(img_path) and os.path.exists(seg_path):
                self.img_paths.append(img_path)
                self.seg_paths.append(seg_path)           
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        """
        Load and return a sample from the dataset.
        Returns:
            tuple: (image, targets) where image is a normalized tensor and targets is a tensor of labels
        """
        # Load image and segmentation mask
        img_path = self.img_paths[idx]
        seg_path = self.seg_paths[idx]
        
        # Open image and segmentation
        image = Image.open(img_path).convert('RGB')
        segmentation = Image.open(seg_path).convert('L')  # Load as grayscale (labels)
        
        # Apply synchronized transforms
        if self.augmentation and self.split == 'train':
            # Random resize and crop
            resize_scale = random.uniform(0.75, 1.25)
            new_size = tuple(int(dim * resize_scale) for dim in self.image_size)
            image = TF.resize(image, new_size, Image.BILINEAR)
            segmentation = TF.resize(segmentation, new_size, Image.NEAREST)
            
            # Random crop to target size
            if new_size[0] > self.image_size[0] and new_size[1] > self.image_size[1]:
                i, j, h, w = transforms.RandomCrop.get_params(
                    image, output_size=self.image_size
                )
                image = TF.crop(image, i, j, h, w)
                segmentation = TF.crop(segmentation, i, j, h, w)
            else:
                # If smaller, resize to exact size
                image = TF.resize(image, self.image_size, Image.BILINEAR)
                segmentation = TF.resize(segmentation, self.image_size, Image.NEAREST)
            
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                segmentation = TF.hflip(segmentation)
            
            # Color jittering (only for image)
            color_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
            image = color_jitter(image)
            
            # Random rotation
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                image = TF.rotate(image, angle, Image.BILINEAR, fill=0)
                segmentation = TF.rotate(segmentation, angle, Image.NEAREST, fill=255)
        else:
            # For validation/test, just resize
            image = TF.resize(image, self.image_size, Image.BILINEAR)
            segmentation = TF.resize(segmentation, self.image_size, Image.NEAREST)
        
        # Convert to tensor
        image = TF.to_tensor(image)
        segmentation = torch.from_numpy(np.array(segmentation, dtype=np.int64))
        
        # Normalize image
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        
        # Ensure invalid labels are set to 255 (ignore index)
        # Valid labels are 0-18, anything else becomes 255
        segmentation[(segmentation < 0) | (segmentation >= 19)] = 255
        
        return image, segmentation


class FaceParsingDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Face Parsing task.
    Handles data loading and preprocessing for training, validation, and testing.
    """
    
    def __init__(
        self,
        data_dir: str = './data',
        image_size: Tuple[int, int] = (512, 512),
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        augmentation: bool = True,
        train_val_split: float = 0.9
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augmentation = augmentation
        self.train_val_split = train_val_split
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        """
        Download or prepare data. Called only on 1 GPU/process.
        """
        # Implement data download/preparation logic here if needed
        pass
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each split.
        Called on every GPU/process.
        """
        if stage == 'fit' or stage is None:
            # Setup training dataset
            self.train_dataset = FaceParsingDataset(
                data_dir=self.data_dir,
                split='train',
                image_size=self.image_size,
                augmentation=self.augmentation
            )
            
            # Setup validation dataset
            self.val_dataset = FaceParsingDataset(
                data_dir=self.data_dir,
                split='val',
                image_size=self.image_size,
                augmentation=False
            )
        
        if stage == 'test' or stage is None:
            # Setup test dataset
            self.test_dataset = FaceParsingDataset(
                data_dir=self.data_dir,
                split='test',
                image_size=self.image_size,
                augmentation=False
            )
    
    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def get_num_classes(self):
        """Return the number of classes in the dataset."""
        # For CelebAMask-HQ dataset
        return 19