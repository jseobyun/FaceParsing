import os
import cv2
import argparse
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision import transforms

from src.models import FaceParsingModel
from src.utils import FaceParsingVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Face Parsing Inference')
    
    parser.add_argument('--input_dir', type=str, default="/media/jseob/HUMAN_3D-PHOTO-01/hairstyle/images/0314.CP138295_none",
                        help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Path to output directory')
    parser.add_argument('--checkpoint', type=str, default="experiments/checkpoints/decoder19.ckpt",
                        help='Path to decoder checkpoint (without dinov3)')
    parser.add_argument('--dinov3_checkpoint', type=str, default="checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
                        help='Path to dinov3 checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512],
                        help='Input image size (height, width)')
    parser.add_argument('--num_classes', type=int, default=19,
                        help='Number of segmentation classes')
    parser.add_argument('--save_overlay', action='store_true', default=True,
                        help='Save overlay visualization')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Transparency for overlay visualization')
    
    return parser.parse_args()


class FaceParsingInference:
    """Class for running face parsing inference."""
    
    def __init__(
        self,
        checkpoint_path: str,
        dinov3_checkpoint_path: str = None,
        device: str = 'cuda',
        image_size: tuple = (512, 512),
        num_classes: int = 19
    ):
        self.device = torch.device(device)
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Load model
        self.model = self._load_model(checkpoint_path, dinov3_checkpoint_path)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Visualizer
        self.visualizer = FaceParsingVisualizer()
    
    def _load_model(self, checkpoint_path: str, dinov3_checkpoint_path: str = None):
        """Load model from checkpoint with separate dinov3 loading."""
        # Load dinov3 model first
        if dinov3_checkpoint_path:
            print(f"Loading DINOv3 from {dinov3_checkpoint_path}")
            REPO_DIR = "src/models/"
            dinov3 = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local',
                                   weights=dinov3_checkpoint_path)
            dinov3 = dinov3.to(self.device)
        else:
            dinov3 = None
            
        # Load decoder checkpoint
        print(f"Loading decoder weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        model = FaceParsingModel(num_classes=self.num_classes)
        
        # Set dinov3 in encoder if provided
        if dinov3 is not None:
            model.encoder.dinov3 = dinov3
            model.encoder.dinov3.eval()
            for p in model.encoder.dinov3.parameters():
                p.requires_grad = False
        
        # Load state dict (decoder and CNN encoder weights)
        if 'state_dict' in checkpoint:
            # Filter out any remaining dinov3 keys if they exist
            state_dict = {k: v for k, v in checkpoint['state_dict'].items() 
                         if 'encoder.dinov3' not in k}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(self, image_path: str):
        """Preprocess input image."""
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Apply transforms
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor, original_size
    
    def predict(self, image_paths):
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Predicted segmentation mask
        """
        # Preprocess image

        image_tensors = []
        for image_path in image_paths:
            image_tensor, original_size = self.preprocess_image(image_path)
            image_tensors.append(image_tensor)
        
        image_tensors = torch.cat(image_tensors, dim=0)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(image_tensors)["probabilities"]
            pred_mask = torch.argmax(logits, dim=1)
        
        # Resize to original size
        # pred_mask = torch.nn.functional.interpolate(
        #     pred_mask.unsqueeze(1).float(),
        #     size=original_size[::-1],  # PIL uses (W, H), torch uses (H, W)
        #     mode='nearest'
        # ).squeeze(1).long()
        
        return pred_mask
    
    def save_results(
        self,
        image_paths,
        pred_masks: torch.Tensor,        
        output_dir: str,
        save_overlay: bool = True,
        alpha: float = 0.5
    ):
        """Save prediction results."""        
        
        os.makedirs(output_dir, exist_ok=True)
        # Get base name
        for img_idx, image_path in enumerate(image_paths):
            base_name = Path(image_path).stem
            
            # Save segmentation mask
            mask_np = pred_masks[img_idx].squeeze().cpu().numpy()
            mask_cv = mask_np.astype(np.uint8)
            # cv2.imwrite(os.path.join(output_dir, base_name+".jpg"), mask_cv)

            # mask_color = self.visualizer.mask_to_colormap(mask_np)        
            # mask_image = Image.fromarray(mask_color)
            # mask_image.save(output_dir / f"{base_name}_mask.png")
            
            # Save overlay if requested
            if save_overlay:
                vis_array = self.visualizer.visualize_prediction(
                    image_path,
                    pred_masks[img_idx].squeeze(),
                    alpha=alpha,
                    save_path=os.path.join(output_dir, base_name+"_vis.jpg")
                )
            
            # print(f"Results saved to {output_dir}")
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        save_overlay: bool = True,
        alpha: float = 0.5,
        batch_size = 1,

    ):
        """Process all images in a directory."""
        input_path = Path(input_dir)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Get all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images to process")
        print(f"These will be processed with batch size {batch_size}")       
       

        num_imgs = len(image_files)

        # Process each image
        for img_idx in range(0, num_imgs, batch_size):
            image_paths = image_files[img_idx:img_idx+batch_size]
            pred_masks = self.predict(image_paths)
            self.save_results(
                image_paths,
                pred_masks,                
                output_dir,
                save_overlay,
                alpha
            )
            


def main():
    """Main inference function."""
    args = parse_args()
    
    # Create inference object
    inference = FaceParsingInference(
        checkpoint_path=args.checkpoint,
        dinov3_checkpoint_path=args.dinov3_checkpoint,
        device=args.device,
        image_size=tuple(args.image_size),
        num_classes=args.num_classes
    )
    
    # Check if input is file or directory


    # Process directory
    inference.process_directory(
        args.input_dir,
        args.output_dir,
        args.save_overlay,
        args.alpha,
        batch_size=16,
    )



if __name__ == '__main__':
    main()