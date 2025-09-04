import torch
import numpy as np
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from PIL import Image
import cv2


class FaceParsingVisualizer:
    """Utility class for visualizing face parsing results."""
    
    # CelebAMask-HQ color palette
    PALETTE = [
        [0, 0, 0],       # background
        [204, 0, 0],     # skin
        [76, 153, 0],    # l_brow
        [204, 204, 0],   # r_brow
        [51, 51, 255],   # l_eye
        [204, 0, 204],   # r_eye
        [0, 255, 255],   # eye_g
        [255, 204, 204], # l_ear
        [102, 51, 0],    # r_ear
        [255, 0, 0],     # ear_r
        [102, 204, 0],   # nose
        [255, 255, 0],   # mouth
        [0, 0, 153],     # u_lip
        [0, 0, 204],     # l_lip
        [255, 51, 255],  # neck
        [0, 204, 204],   # neck_l
        [0, 51, 0],      # cloth
        [255, 153, 51],  # hair
        [0, 204, 0]      # hat
    ]
    
    CLASS_NAMES = [
        'background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
        'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth',
        'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat'
    ]
    
    @staticmethod
    def mask_to_colormap(mask: np.ndarray, num_classes: int = 19) -> np.ndarray:
        """
        Convert segmentation mask to color image.
        
        Args:
            mask: Segmentation mask of shape (H, W)
            num_classes: Number of classes
            
        Returns:
            Color image of shape (H, W, 3)
        """
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for label_id in range(num_classes):
            color_mask[mask == label_id] = FaceParsingVisualizer.PALETTE[label_id]
        
        return color_mask
    
    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert torch tensor to numpy array.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Numpy array
        """
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Handle different tensor shapes
        if len(tensor.shape) == 4:  # Batch dimension
            tensor = tensor[0]
        
        if len(tensor.shape) == 3:  # Channel dimension
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            elif tensor.shape[0] == 3:
                # Denormalize if needed (ImageNet normalization)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                tensor = tensor * std + mean
                tensor = tensor.clamp(0, 1)
                tensor = tensor.permute(1, 2, 0)
        
        return tensor.numpy()
    
    @staticmethod
    def visualize_prediction(
        image: torch.Tensor,
        pred_mask: torch.Tensor,
        gt_mask: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize prediction results.
        
        Args:
            image: Input image tensor
            pred_mask: Predicted mask tensor
            gt_mask: Ground truth mask tensor (optional)
            alpha: Transparency for overlay
            save_path: Path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        # Convert to numpy
        image_np = FaceParsingVisualizer.tensor_to_numpy(image)
        pred_np = FaceParsingVisualizer.tensor_to_numpy(pred_mask)
        
        # Convert to uint8
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)
        
        # Get color masks
        pred_color = FaceParsingVisualizer.mask_to_colormap(pred_np.astype(np.int32))
        
        # Create overlay
        overlay = cv2.addWeighted(image_np, 1 - alpha, pred_color, alpha, 0)
        
        # Create figure
        if gt_mask is not None:
            gt_np = FaceParsingVisualizer.tensor_to_numpy(gt_mask)
            gt_color = FaceParsingVisualizer.mask_to_colormap(gt_np.astype(np.int32))
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(image_np)
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            axes[1].imshow(gt_color)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            axes[2].imshow(pred_color)
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            axes[3].imshow(overlay)
            axes[3].set_title('Overlay')
            axes[3].axis('off')
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(image_np)
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            axes[1].imshow(pred_color)
            axes[1].set_title('Prediction')
            axes[1].axis('off')
            
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        # Convert figure to numpy array
        fig.canvas.draw()
        vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        
        return vis_array
    
    @staticmethod
    def create_legend(num_classes: int = 19, save_path: Optional[str] = None) -> np.ndarray:
        """
        Create a legend for the segmentation classes.
        
        Args:
            num_classes: Number of classes
            save_path: Path to save legend
            
        Returns:
            Legend as numpy array
        """
        fig, ax = plt.subplots(figsize=(4, 6))
        
        for i in range(num_classes):
            color = np.array(FaceParsingVisualizer.PALETTE[i]) / 255.0
            ax.add_patch(plt.Rectangle((0, i), 1, 1, color=color))
            ax.text(1.2, i + 0.5, FaceParsingVisualizer.CLASS_NAMES[i],
                   va='center', fontsize=10)
        
        ax.set_xlim(0, 3)
        ax.set_ylim(0, num_classes)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Face Parsing Classes', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        # Convert to numpy array
        fig.canvas.draw()
        legend_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        legend_array = legend_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        
        return legend_array