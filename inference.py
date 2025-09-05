import argparse
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms

from src.models import FaceParsingModel
from src.utils import FaceParsingVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Face Parsing Inference')
    
    parser.add_argument('--input_dir', type=str, default="/media/jseob/X9/renderme360/processed/0609/e0/frames/000",
                        help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Path to output directory')
    parser.add_argument('--checkpoint', type=str, default="experiments/checkpoints/last-v1.ckpt",
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512],
                        help='Input image size (height, width)')
    parser.add_argument('--num_classes', type=int, default=19,
                        help='Number of segmentation classes')
    parser.add_argument('--save_overlay', action='store_true',
                        help='Save overlay visualization')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Transparency for overlay visualization')
    
    return parser.parse_args()


class FaceParsingInference:
    """Class for running face parsing inference."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        image_size: tuple = (512, 512),
        num_classes: int = 19
    ):
        self.device = torch.device(device)
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Visualizer
        self.visualizer = FaceParsingVisualizer()
    
    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        model = FaceParsingModel(num_classes=self.num_classes)
        
        # Load state dict
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
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
    
    def predict(self, image_path: str):
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Predicted segmentation mask
        """
        # Preprocess image
        image_tensor, original_size = self.preprocess_image(image_path)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(image_tensor)["probabilities"]
            pred_mask = torch.argmax(logits, dim=1)
        
        # Resize to original size
        pred_mask = torch.nn.functional.interpolate(
            pred_mask.unsqueeze(1).float(),
            size=original_size[::-1],  # PIL uses (W, H), torch uses (H, W)
            mode='nearest'
        ).squeeze(1).long()
        
        return pred_mask, image_tensor
    
    def save_results(
        self,
        image_path: str,
        pred_mask: torch.Tensor,
        image_tensor: torch.Tensor,
        output_dir: str,
        save_overlay: bool = True,
        alpha: float = 0.5
    ):
        """Save prediction results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get base name
        base_name = Path(image_path).stem
        
        # Save segmentation mask
        mask_np = pred_mask.squeeze().cpu().numpy()
        mask_color = self.visualizer.mask_to_colormap(mask_np)
        mask_image = Image.fromarray(mask_color)
        mask_image.save(output_dir / f"{base_name}_mask.png")
        
        # Save overlay if requested
        if save_overlay:
            vis_array = self.visualizer.visualize_prediction(
                image_tensor.squeeze(),
                pred_mask.squeeze(),
                alpha=alpha,
                save_path=str(output_dir / f"{base_name}_visualization.png")
            )
        
        print(f"Results saved to {output_dir}")
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        save_overlay: bool = True,
        alpha: float = 0.5
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
        
        # Process each image
        for image_path in image_files:
            print(f"Processing {image_path.name}...")
            
            
            pred_mask, image_tensor = self.predict(str(image_path))
            self.save_results(
                str(image_path),
                pred_mask,
                image_tensor,
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
        device=args.device,
        image_size=tuple(args.image_size),
        num_classes=args.num_classes
    )
    
    # Check if input is file or directory
    input_path = Path(args.input_dir)
    
    if input_path.is_file():
        # Process single image
        print(f"Processing image: {input_path}")
        pred_mask, image_tensor = inference.predict(str(input_path))
        inference.save_results(
            str(input_path),
            pred_mask,
            image_tensor,
            args.output_dir,
            args.save_overlay,
            args.alpha
        )
    elif input_path.is_dir():
        # Process directory
        inference.process_directory(
            str(input_path),
            args.output_dir,
            args.save_overlay,
            args.alpha
        )
    else:
        raise ValueError(f"Input path {input_path} does not exist")


if __name__ == '__main__':
    main()