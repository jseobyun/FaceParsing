import argparse
import torch

from src.face_parsing import FaceParsingPredictor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Face Parsing Inference")

    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/jseob/Desktop/yjs/data/reactor_test/source",
        help="Path to input image or directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Path to output directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/checkpoints/decoder.ckpt",
        help="Path to decoder checkpoint (without DINOv3)",
    )
    parser.add_argument(
        "--dinov3_checkpoint",
        type=str,
        default="checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        help="Path to DINOv3 checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Input image size (height, width)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of images processed per batch",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Transparency for overlay visualization",
    )
    parser.add_argument(
        "--save_overlay",
        action="store_true",
        dest="save_overlay",
        help="Save overlay visualization (default: True)",
    )
    parser.add_argument(
        "--no_overlay",
        action="store_false",
        dest="save_overlay",
        help="Disable overlay visualization",
    )
    parser.set_defaults(save_overlay=True)

    return parser.parse_args()


def main():
    """Entry point for CLI-based inference."""
    args = parse_args()

    predictor = FaceParsingPredictor(
        checkpoint_path=args.checkpoint,
        dinov3_checkpoint_path=args.dinov3_checkpoint,
        device=args.device,
        image_size=tuple(args.image_size),
    )

    predictor.predict_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        save_overlay=args.save_overlay,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
