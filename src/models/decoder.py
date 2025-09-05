import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ConvBNReLU(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding, dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficiency."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class AttentionRefinementModule(nn.Module):
    """Attention refinement module for feature enhancement."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.attention(x)
        return x * attention


class FeatureFusionModule(nn.Module):
    """Feature fusion module for combining multi-scale features."""
    
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, out_channels, 1, 1, 0)
        
        # Channel attention
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(out_channels, out_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(out_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x2 is not None:
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
            
        x = self.conv(x)
        
        # Apply channel attention
        gap = self.gap(x)
        attention = self.fc1(gap)
        attention = self.relu(attention)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention)
        
        return x * attention


class CoarseToFineBlock(nn.Module):
    """Coarse-to-fine refinement block."""
    
    def __init__(self, coarse_channels: int, fine_channels: int, out_channels: int):
        super().__init__()
        self.coarse_conv = ConvBNReLU(coarse_channels, out_channels, 3, 1, 1)
        self.fine_conv = ConvBNReLU(fine_channels, out_channels, 3, 1, 1)
        self.fusion = FeatureFusionModule(out_channels * 2, out_channels)
        self.refine = nn.Sequential(
            DepthwiseSeparableConv(out_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels)
        )
        
    def forward(self, coarse: torch.Tensor, fine: torch.Tensor) -> torch.Tensor:
        # Process coarse features
        coarse = self.coarse_conv(coarse)
        # Upsample to match fine resolution
        coarse_up = F.interpolate(coarse, size=fine.shape[2:], 
                                 mode='bilinear', align_corners=False)
        
        # Process fine features
        fine = self.fine_conv(fine)
        
        # Fuse features
        fused = self.fusion(coarse_up, fine)
        
        # Refine
        refined = self.refine(fused)
        
        return refined


class Decoder(nn.Module):
    """
    Coarse-to-fine decoder for face parsing segmentation.
    Designed to work with VGG19 + DINOv3 encoder output.
    
    The encoder outputs a feature pyramid with scales: {1, 2, 4, 8, 16}
    - Scale 1: VGG19 features (channels: 64)
    - Scale 2: VGG19 features (channels: 128)
    - Scale 4: VGG19 features (channels: 256)
    - Scale 8: VGG19 features (channels: 512)
    - Scale 16: DINOv3 features (channels: 1024)
    """
    
    def __init__(self, num_classes: int = 11, decoder_channels: int = 256):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Channel adaptation layers for each scale
        # VGG19 channels at different scales (before MaxPool)
        self.adapt_scale1 = ConvBNReLU(64, decoder_channels // 4, 1, 1, 0)   # scale 1 (64 channels)
        self.adapt_scale2 = ConvBNReLU(128, decoder_channels // 2, 1, 1, 0)  # scale 2 (128 channels)
        self.adapt_scale4 = ConvBNReLU(256, decoder_channels, 1, 1, 0)       # scale 4 (256 channels)
        self.adapt_scale8 = ConvBNReLU(512, decoder_channels, 1, 1, 0)       # scale 8 (512 channels)
        self.adapt_scale16 = ConvBNReLU(1024, decoder_channels * 2, 1, 1, 0) # scale 16 (DINOv3, 1024 channels)
        
        # Attention refinement for DINOv3 features
        self.arm16 = AttentionRefinementModule(decoder_channels * 2)
        
        # Coarse-to-fine blocks
        self.ctf_16_8 = CoarseToFineBlock(decoder_channels * 2, decoder_channels, decoder_channels)
        self.ctf_8_4 = CoarseToFineBlock(decoder_channels, decoder_channels, decoder_channels)
        self.ctf_4_2 = CoarseToFineBlock(decoder_channels, decoder_channels // 2, decoder_channels // 2)
        self.ctf_2_1 = CoarseToFineBlock(decoder_channels // 2, decoder_channels // 4, decoder_channels // 4)
        
        # Feature fusion modules for skip connections
        self.ffm_8 = FeatureFusionModule(decoder_channels * 2, decoder_channels)
        self.ffm_4 = FeatureFusionModule(decoder_channels * 2, decoder_channels)
        self.ffm_2 = FeatureFusionModule(decoder_channels, decoder_channels // 2)
        self.ffm_1 = FeatureFusionModule(decoder_channels // 2, decoder_channels // 4)
        
        # Auxiliary heads for deep supervision (optional)
        self.aux_head_8 = nn.Conv2d(decoder_channels, num_classes, 1)
        self.aux_head_4 = nn.Conv2d(decoder_channels, num_classes, 1)
        self.aux_head_2 = nn.Conv2d(decoder_channels // 2, num_classes, 1)
        
        # Final segmentation head
        self.final_conv = nn.Sequential(
            ConvBNReLU(decoder_channels // 4, decoder_channels // 4, 3, 1, 1),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels // 4, num_classes, 1)
        )
        
    def forward(self, feature_pyramid: Dict[int, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the decoder.
        
        Args:
            feature_pyramid: Dictionary with scales as keys (1, 2, 4, 8, 16) 
                           and features as values
                           
        Returns:
            Dictionary containing:
                - 'segmentation': Final segmentation logits (B, num_classes, H, W)
                - 'aux_outputs': Dictionary of auxiliary outputs for deep supervision
                - 'one_hot': One-hot encoded segmentation (B, num_classes, H, W)
        """
        
        # Extract and adapt features at each scale
        f1 = self.adapt_scale1(feature_pyramid[1])    # Finest
        f2 = self.adapt_scale2(feature_pyramid[2])
        f4 = self.adapt_scale4(feature_pyramid[4])
        f8 = self.adapt_scale8(feature_pyramid[8])
        f16 = self.adapt_scale16(feature_pyramid[16])  # Coarsest (DINOv3)
        
        # Apply attention to DINOv3 features
        f16 = self.arm16(f16)
        
        # Coarse-to-fine refinement with skip connections
        # Stage 16 -> 8
        refined_8 = self.ctf_16_8(f16, f8)
        refined_8 = self.ffm_8(refined_8, f8)
        
        # Stage 8 -> 4
        refined_4 = self.ctf_8_4(refined_8, f4)
        refined_4 = self.ffm_4(refined_4, f4)
        
        # Stage 4 -> 2
        refined_2 = self.ctf_4_2(refined_4, f2)
        refined_2 = self.ffm_2(refined_2, f2)
        
        # Stage 2 -> 1
        refined_1 = self.ctf_2_1(refined_2, f1)
        refined_1 = self.ffm_1(refined_1, f1)
        
        # Generate auxiliary outputs for deep supervision
        aux_outputs = {}
        if self.training:
            aux_8 = self.aux_head_8(refined_8)
            aux_4 = self.aux_head_4(refined_4)
            aux_2 = self.aux_head_2(refined_2)
            
            # Upsample to original resolution
            original_size = feature_pyramid[1].shape[2:]
            aux_outputs['aux_8'] = F.interpolate(aux_8, size=original_size, 
                                                mode='bilinear', align_corners=False)
            aux_outputs['aux_4'] = F.interpolate(aux_4, size=original_size,
                                                mode='bilinear', align_corners=False)
            aux_outputs['aux_2'] = F.interpolate(aux_2, size=original_size,
                                                mode='bilinear', align_corners=False)
        
        # Final segmentation
        segmentation = self.final_conv(refined_1) # B num_parts 512 512
        
        # Generate one-hot encoding
        segmentation_softmax = F.softmax(segmentation, dim=1)
        _, predicted_classes = torch.max(segmentation_softmax, dim=1, keepdim=True)
        one_hot = torch.zeros_like(segmentation)
        one_hot.scatter_(1, predicted_classes, 1)
        
        return {
            'segmentation': segmentation,
            'aux_outputs': aux_outputs,
            'one_hot': one_hot,
            'probabilities': segmentation_softmax
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                     targets: torch.Tensor, 
                     aux_weight: float = 0.4) -> Dict[str, torch.Tensor]:
        """
        Compute segmentation loss with deep supervision using BCE.
        
        Args:
            outputs: Dictionary from forward pass
            targets: Ground truth segmentation one-hot or soft labels (B, num_classes, H, W)
                    or class indices (B, H, W) which will be converted to one-hot
            aux_weight: Weight for auxiliary losses
            
        Returns:
            Dictionary with total loss and components
        """
        # Convert targets to one-hot if needed
        if targets.dim() == 3:  # (B, H, W) - class indices
            targets_one_hot = torch.zeros_like(outputs['segmentation'])
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        else:  # Already one-hot or soft labels
            targets_one_hot = targets
        
        # Use BCE with logits for stability
        criterion = nn.BCEWithLogitsLoss()
        
        # Main segmentation loss
        main_loss = criterion(outputs['segmentation'], targets_one_hot.float())
        
        # Auxiliary losses for deep supervision
        total_loss = main_loss
        aux_loss = 0
        
        if 'aux_outputs' in outputs and outputs['aux_outputs']:
            for aux_out in outputs['aux_outputs'].values():
                aux_loss += criterion(aux_out, targets_one_hot.float())
            
            aux_loss = aux_loss / len(outputs['aux_outputs'])
            total_loss = main_loss + aux_weight * aux_loss
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'aux_loss': aux_loss if isinstance(aux_loss, torch.Tensor) else torch.tensor(aux_loss)
        }

if __name__ == "__main__":
    # Test decoder with simulated encoder output
    batch_size = 2
    H, W = 512, 512
    
    # Simulate feature pyramid from VGG19 + DINOv3 encoder
    feature_pyramid = {
        1: torch.randn(batch_size, 64, H, W),       # VGG19 scale 1
        2: torch.randn(batch_size, 128, H//2, W//2), # VGG19 scale 2
        4: torch.randn(batch_size, 256, H//4, W//4), # VGG19 scale 4
        8: torch.randn(batch_size, 512, H//8, W//8), # VGG19 scale 8
        16: torch.randn(batch_size, 1024, H//16, W//16), # DINOv3 scale 16
    }
    
    # Create decoder
    decoder = Decoder(num_classes=11)
    
    # Set to eval mode for testing
    decoder.eval()
    
    # Forward pass
    with torch.no_grad():
        outputs = decoder(feature_pyramid)
    
    print(f"Segmentation shape: {outputs['segmentation'].shape}")
    print(f"One-hot shape: {outputs['one_hot'].shape}")
    print(f"Probabilities shape: {outputs['probabilities'].shape}")
    
    # Test with training mode for auxiliary outputs
    decoder.train()
    outputs = decoder(feature_pyramid)
    print(f"Auxiliary outputs: {list(outputs['aux_outputs'].keys())}")
    
    # Test loss computation with class indices
    targets = torch.randint(0, 19, (batch_size, H, W))
    losses = decoder.compute_loss(outputs, targets)
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    print(f"Main loss: {losses['main_loss'].item():.4f}")
    print(f"Aux loss: {losses['aux_loss'].item():.4f}")
    
    # Test with one-hot targets
    targets_one_hot = torch.zeros(batch_size, 19, H, W)
    targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
    losses = decoder.compute_loss(outputs, targets_one_hot)
    print(f"\nWith one-hot targets:")
    print(f"Total loss: {losses['total_loss'].item():.4f}")