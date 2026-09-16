import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import os
import zipfile
from PIL import Image

# UNet Model for DDPM
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x, t_emb):
        h = self.conv1(F.relu(x))
        h = self.norm1(h)
        t = self.time_mlp(F.relu(t_emb))[:, :, None, None]
        h = h + t
        h = self.conv2(F.relu(h))
        h = self.norm2(h)
        return h + self.skip(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_dim=256, base_channels=96):
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        # Encoder (2 ResBlocks per level)
        self.enc1_1 = ResBlock(in_channels, base_channels, time_dim)
        self.enc1_2 = ResBlock(base_channels, base_channels, time_dim)
        
        self.enc2_1 = ResBlock(base_channels, base_channels * 2, time_dim)
        self.enc2_2 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        
        self.enc3_1 = ResBlock(base_channels * 2, base_channels * 4, time_dim)
        self.enc3_2 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        
        # Bottleneck
        self.bottleneck = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        
        # Decoder (2 ResBlocks per level)
        self.dec3_1 = ResBlock(base_channels * 8, base_channels * 4, time_dim)
        self.dec3_2 = ResBlock(base_channels * 4, base_channels * 2, time_dim)
        
        self.dec2_1 = ResBlock(base_channels * 4, base_channels * 2, time_dim)
        self.dec2_2 = ResBlock(base_channels * 2, base_channels, time_dim)
        
        self.dec1_1 = ResBlock(base_channels * 2, base_channels, time_dim)
        self.dec1_2 = ResBlock(base_channels, base_channels, time_dim)
        
        self.out = nn.Conv2d(base_channels, out_channels, 1)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        # Encoder
        e1 = self.enc1_1(x, t_emb)
        e1 = self.enc1_2(e1, t_emb)  # 28x28
        
        e2 = self.enc2_1(self.pool(e1), t_emb)
        e2 = self.enc2_2(e2, t_emb)  # 14x14
        
        e3 = self.enc3_1(self.pool(e2), t_emb)
        e3 = self.enc3_2(e3, t_emb)  # 7x7
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3), t_emb)  # 3x3
        
        # Decoder with size matching
        d3 = F.interpolate(b, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3_1(torch.cat([d3, e3], dim=1), t_emb)
        d3 = self.dec3_2(d3, t_emb)  # 7x7
        
        d2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2_1(torch.cat([d2, e2], dim=1), t_emb)
        d2 = self.dec2_2(d2, t_emb)  # 14x14
        
        d1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1_1(torch.cat([d1, e1], dim=1), t_emb)
        d1 = self.dec1_2(d1, t_emb)  # 28x28
        
        return self.out(d1)

class DDPM:
    def __init__(self, model, timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda', schedule='cosine'):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # Cosine schedule
        self.betas = self.cosine_beta_schedule(timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
        
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:, None, None, None]
        
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None, None, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, batch_size, channels=3, height=28, width=28):
        self.model.eval()
        x = torch.randn(batch_size, channels, height, width).to(self.device)
        
        for i in tqdm(reversed(range(self.timesteps)), desc='Sampling', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, i)
        
        return x

class EMA:
    """
    Exponential Moving Average for model parameters
    """
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def extract_mnist_zip(zip_path='mnist.zip', extract_to='./data'):
    """
    Extract MNIST dataset from zip file
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"MNIST zip file not found at {zip_path}")
    
    print(f"Extracting {zip_path}...")
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"Extraction completed to {extract_to}/")
    
    # Count extracted images
    png_files = [f for f in os.listdir(extract_to) if f.endswith('.png')]
    print(f"Found {len(png_files)} PNG images")

class MNISTPNGDataset(Dataset):
    """
    Custom Dataset for MNIST PNG images (00001.png ~ 60000.png)
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Get all PNG files
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
        
        if len(self.image_files) == 0:
            raise FileNotFoundError(f"No PNG images found in {data_dir}")
        
        print(f"Loaded {len(self.image_files)} images from {data_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path)
        
        # Convert grayscale to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Return image and dummy label (not used in DDPM)
        return image, 0

def train_ddpm(epochs=300, batch_size=128, lr=2e-4, save_dir='checkpoints', mnist_zip_path='mnist.zip', data_dir='./mnist', base_channels=96, schedule='cosine'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract MNIST zip file if it exists
    if os.path.exists(mnist_zip_path):
        extract_mnist_zip(mnist_zip_path, extract_to=data_dir)
    else:
        print(f"Warning: {mnist_zip_path} not found. Using existing data in {data_dir}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transformation (images are already 28x28 RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    # Load PNG dataset
    dataset = MNISTPNGDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Model
    model = UNet(base_channels=base_channels).to(device)
    ddpm = DDPM(model, timesteps=1000, device=device, schedule=schedule)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # EMA
    ema = EMA(model, decay=0.9999)

    # Training
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            
            t = torch.randint(0, ddpm.timesteps, (images.shape[0],), device=device).long()
            loss = ddpm.p_losses(images, t)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update EMA
            ema.update()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'ema_shadow': ema.shadow,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model with EMA weights
    ema.apply_shadow()
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    print("Training completed!")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DDPM on MNIST')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--mnist_zip', type=str, default='mnist.zip', help='Path to MNIST zip file')
    parser.add_argument('--data_dir', type=str, default='./mnist', help='Directory to extract/load images')
    parser.add_argument('--base_channels', type=int, default=96, help='Base number of channels in UNet')
    parser.add_argument('--schedule', type=str, default='cosine', help='Noise schedule')
    args = parser.parse_args()
    
    train_ddpm(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        mnist_zip_path=args.mnist_zip,
        data_dir=args.data_dir,
        base_channels=args.base_channels,
        schedule=args.schedule
    )