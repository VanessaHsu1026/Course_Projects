import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from train import UNet, DDPM

def visualize_diffusion_process(model_path, save_path='diffusion_process.png'):
    """
    Visualize the diffusion process for 8 samples at 8 different timesteps
    Creates an 8x8 grid as required in the assignment
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    ddpm = DDPM(model, timesteps=1000, device=device)
    
    num_samples = 8
    num_steps = 8
    
    # Calculate timestep intervals (divide into 7 equal parts = 8 points)
    # Example: 1000 steps / 7 = ~142 steps apart
    timestep_interval = ddpm.timesteps // (num_steps - 1)
    timesteps_to_save = [i * timestep_interval for i in range(num_steps - 1)]
    timesteps_to_save.append(ddpm.timesteps - 1)  # Add the last step
    timesteps_to_save.reverse()  # Reverse to go from noisy to clean
    
    print(f"Recording timesteps: {timesteps_to_save}")
    
    # Store intermediate results
    all_samples = []
    
    # Generate samples
    with torch.no_grad():
        x = torch.randn(num_samples, 3, 28, 28).to(device)
        
        samples_at_steps = {step: None for step in timesteps_to_save}
        step_idx = 0
        
        for i in range(ddpm.timesteps - 1, -1, -1):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            
            # Record at specific timesteps
            if i in timesteps_to_save:
                # Denormalize and save
                img = (x + 1) / 2
                img = torch.clamp(img, 0, 1)
                samples_at_steps[i] = img.cpu().numpy()
            
            # Denoise
            betas_t = ddpm.betas[t][:, None, None, None]
            sqrt_one_minus_alphas_cumprod_t = ddpm.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            sqrt_recip_alphas_t = ddpm.sqrt_recip_alphas[t][:, None, None, None]
            
            model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
            )
            
            if i > 0:
                posterior_variance_t = ddpm.posterior_variance[t][:, None, None, None]
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(posterior_variance_t) * noise
            else:
                x = model_mean
    
    # Create visualization (8 rows = 8 timesteps, 8 cols = 8 samples)
    # This matches the homework example layout
    fig, axes = plt.subplots(num_steps, num_samples, figsize=(16, 16))
    fig.patch.set_facecolor('black')

    for step_idx, timestep in enumerate(timesteps_to_save):
        for sample_idx in range(num_samples):
            ax = axes[step_idx, sample_idx]
            img = samples_at_steps[timestep][sample_idx]
            img = np.transpose(img, (1, 2, 0))  # CHW to HWC
            
            ax.imshow(img)
            ax.axis('off')


    plt.subplots_adjust(wspace=0.06, hspace=0.06)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"Diffusion process visualization saved to {save_path}")
    plt.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize DDPM diffusion process')
    parser.add_argument('--model_path', type=str, default='checkpoints/final_model.pth',
                        help='Path to trained model')
    parser.add_argument('--save_path', type=str, default='diffusion_process.png',
                        help='Path to save visualization')
    
    args = parser.parse_args()
    
    visualize_diffusion_process(
        model_path=args.model_path,
        save_path=args.save_path
    )