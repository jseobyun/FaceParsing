import torch
import argparse
from pathlib import Path

def extract_decoder_weights(checkpoint_path: str, output_path: str):
    """
    Extract decoder weights from checkpoint, excluding dinov3 encoder weights.
    
    Args:
        checkpoint_path: Path to the full checkpoint (last.ckpt)
        output_path: Path to save the decoder-only checkpoint
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create new checkpoint without dinov3 weights
    new_checkpoint = {}
    
    # Copy all non-model data
    for key in checkpoint.keys():
        if key != 'state_dict':
            new_checkpoint[key] = checkpoint[key]
    
    # Filter state_dict to exclude dinov3 weights
    new_state_dict = {}
    dinov3_keys_removed = []
    
    for key, value in checkpoint['state_dict'].items():
        # Skip dinov3 related keys
        if 'encoder.dinov3' in key:
            dinov3_keys_removed.append(key)
        else:
            new_state_dict[key] = value
    
    new_checkpoint['state_dict'] = new_state_dict
    
    # Save the new checkpoint
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_checkpoint, output_path)
    
    print(f"\nDecoder checkpoint saved to {output_path}")
    print(f"Removed {len(dinov3_keys_removed)} dinov3 encoder keys")
    print(f"Remaining keys: {len(new_state_dict)}")
    
    # Print some removed keys for verification
    if dinov3_keys_removed:
        print("\nSample of removed dinov3 keys:")
        for key in dinov3_keys_removed[:5]:
            print(f"  - {key}")
        if len(dinov3_keys_removed) > 5:
            print(f"  ... and {len(dinov3_keys_removed) - 5} more")

def main():
    parser = argparse.ArgumentParser(description='Extract decoder weights without dinov3')
    parser.add_argument('--checkpoint', type=str, default='experiments/checkpoints/last.ckpt',
                        help='Path to full checkpoint')
    parser.add_argument('--output', type=str, default='experiments/checkpoints/decoder.ckpt',
                        help='Path to save decoder-only checkpoint')
    
    args = parser.parse_args()
    extract_decoder_weights(args.checkpoint, args.output)

if __name__ == '__main__':
    main()