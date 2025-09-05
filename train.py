import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pathlib import Path

from src.models import FaceParsingModel
from src.data import FaceParsingDataModule


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Face Parsing Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='/media/idc-r2w2/data_sdc/jseob/data/synthface/dataset_100000',
                        help='Path to data directory')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512],
                        help='Input image size (height, width)')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=11,
                        help='Number of segmentation classes')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=12,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--max_epochs', type=int, default=20,
                        help='Maximum number of training epochs')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--accelerator', type=str, default='auto',
                        help='Accelerator to use (auto, gpu, cpu)')
    
    # Experiment arguments
    parser.add_argument('--experiment_name', type=str, default='face_parsing',
                        help='Name of the experiment')
    parser.add_argument('--checkpoint_dir', type=str, default='./experiments/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./experiments/logs',
                        help='Directory to save logs')
    
    # Other arguments
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def setup_callbacks(args):
    """Setup training callbacks."""
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.checkpoint_dir) / args.experiment_name,
        filename='{epoch:02d}-{val_loss:.4f}',
        monitor='val/loss',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val/loss',
        patience=15,
        mode='min',
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    return callbacks


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()    
    
    # Set random seed for reproducibility
    pl.seed_everything(args.seed)
    
    # Create data module
    data_module = FaceParsingDataModule(
        data_dir=args.data_dir,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        augmentation=True# ugly hack to not show parameters to DDP    
    )
    
    # Create model
    model = FaceParsingModel(
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_config={
            'T_max': args.max_epochs,
            'eta_min': 1e-6
        }
    )
    
    # Setup callbacks
    callbacks = setup_callbacks(args)
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
        default_hp_metric=False
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.gpus if args.accelerator == 'gpu' else 'auto',
        callbacks=callbacks,
        logger=logger,
        precision=32,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        log_every_n_steps=10,
        val_check_interval=1.0,
        deterministic=False
    )
    
    # Train model
    if args.resume_from:
        trainer.fit(model, data_module, ckpt_path=args.resume_from)
    else:
        trainer.fit(model, data_module)
    
    # Test model with best checkpoint
    trainer.test(model, data_module, ckpt_path='best')


if __name__ == '__main__':
    main()