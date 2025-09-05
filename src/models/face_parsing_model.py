import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, JaccardIndex
from typing import Dict, Any, Optional
from .encoder import Encoder
from .decoder import Decoder

class FaceParsingModel(pl.LightningModule):
    """
    PyTorch Lightning module for Face Parsing task.
    This is a base implementation that needs to be extended with specific network architecture.
    """
    
    def __init__(
        self,
        num_classes: int = 11,  # Ours has 11 labels
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Network architecture will be defined here
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes, ignore_index=255)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes, ignore_index=255)
        self.val_iou = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255)
        
    def forward(self, x):
        """Forward pass through the network."""
        # Placeholder implementation
        features = self.encoder(x)
        output = self.decoder(features)
        return output
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                     targets: torch.Tensor, 
                     aux_weight: float = 0.4) -> Dict[str, torch.Tensor]:        
        
        
        # Main segmentation loss
        main_loss = self.criterion(outputs["segmentation"], targets.long())
        
        # Auxiliary losses for deep supervision
        total_loss = main_loss
        aux_loss = 0
        
        if 'aux_outputs' in outputs and outputs['aux_outputs']:
            for aux_out in outputs['aux_outputs'].values():
                aux_loss += self.criterion(aux_out, targets.long())
            
            aux_loss = aux_loss / len(outputs['aux_outputs'])
            total_loss = main_loss + aux_weight * aux_loss
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'aux_loss': aux_loss if isinstance(aux_loss, torch.Tensor) else torch.tensor(aux_loss)
        }
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        images, targets = batch
        
        # Forward pass
        outputs = self(images)
        
        # Calculate loss
        loss = self.compute_loss(outputs, targets)                
        self.log("train_loss", loss["total_loss"])
        
        return {"loss" : loss["total_loss"]}
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""        
        images, targets = batch
        
        # Forward pass
        outputs = self(images)
        
        # Calculate loss
        loss = self.compute_loss(outputs, targets)                
        self.log("val/loss", loss["total_loss"])
        self.log("val/main_loss", loss["main_loss"])
        self.log("val/aux_loss", loss["aux_loss"])       
                        
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        if self.hparams.scheduler_config is None:
            return optimizer
        
        # Example scheduler configuration
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.scheduler_config.get('T_max', 100),
            eta_min=self.hparams.scheduler_config.get('eta_min', 1e-6)
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }