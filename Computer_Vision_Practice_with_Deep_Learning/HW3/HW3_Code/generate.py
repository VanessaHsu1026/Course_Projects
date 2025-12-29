import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
import os
from train import UNet, DDPM


def generate_images(model_path, output_dir='generated_images', num_images=10000, batch_size=100):
    """
    Generate images using trained DDPM model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    ddpm = DDPM(model, timesteps=1000, device=device)
    
    # Generate images
    num_batches = (num_images + batch_size - 1) // batch_size
    image_count = 0
    
    print(f"Generating {num_images} images...")
    
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_images - image_count)
        
        # Sample from model
        with torch.no_grad():
            images = ddpm.sample(current_batch_size, channels=3, height=28, width=28)
        
        # Denormalize from [-1, 1] to [0, 1]
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
        
        # Save individual images
        for i in range(current_batch_size):
            image_count += 1
            filename = f"{image_count:05d}.png"
            save_image(images[i], os.path.join(output_dir, filename))
        
        print(f"Generated {image_count}/{num_images} images")
    
    print(f"All images saved to {output_dir}/")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate images using DDPM')
    parser.add_argument('--model_path', type=str, default='checkpoints/final_model.pth',
                        help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='generated_images',
                        help='Directory to save generated images')
    parser.add_argument('--num_images', type=int, default=10000,
                        help='Number of images to generate')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for generation')
    
    args = parser.parse_args()
    
    generate_images(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_images=args.num_images,
        batch_size=args.batch_size
    )