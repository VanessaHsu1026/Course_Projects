---
title: 'HW1: Comparison of AE and VAE'
---

# HW1: Comparison of AE and VAE
##### In this project, I will create an autoencoder (AE) and a variational autoencoder (VAE) using PyTorch, training them on eye datasets to reconstruct images. I will evaluate the models by calculating the average PSNR and average SSIM for quantitative and qualitative comparisons. On the other hand, I will introduce Gaussian noise into both the autoencoder (AE) and the variational autoencoder (VAE) during image reconstruction, and then compare the reconstruction results with and without the noise.

## Environment
###### Step 1. Click the share link of  HW1_113064525_徐綉惠.zip
###### Step 2. Download HW1_113064525_徐綉惠.zip
###### Step 3. Upload HW1_113064525_徐綉惠 folder to your Google Drive
###### Step 4. Upload eye.zip to HW1_113064525_徐綉惠 folder
###### Step 5. Open HW1.ipynb
###### Step 6. Please execute each cell sequentially from top to bottom

### 1. Unzip eye dataset
```python
from google.colab import drive
drive.mount('/content/drive')

!ls /content/drive/
%cd "/content/drive/My Drive/HW1_113064525_徐綉惠/"
!unzip eye.zip -d "./eye_dataset/"
!ls "./eye_dataset/"
```

### 2. Import libraries
```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import random
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
from PIL import Image
```

### 3. Custom dataset class
```python
class Eye_Dataset(Dataset):
    def __init__(self, data_path, label_path, transform=None):
        # load data and labels
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
        self.transform = transform

    def __len__(self):
        # the total number of images
        return len(self.data)

    def __getitem__(self, idx):
        # get the image and label at the specified index
        image = self.data[idx]
        label = self.labels[idx]

        # apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

# define the transformation of images
transform = transforms.Compose([
    # convert NumPy array to tensor and normalize to [0, 1]
    transforms.ToTensor(),
    # convert the data type from float64 to float32
    transforms.ConvertImageDtype(torch.float32)
])

# PyTorch Dataset Class
dataset = Eye_Dataset(data_path='./eye_dataset/eye/data.npy', label_path='./eye_dataset/eye/label.npy', transform=transform)

# set the seed for reproducibility
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
```

### 4. CNN-Based AE and VAE
```python
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Output: 16 x 25 x 25
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Output: 32 x 13 x 13
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Output: 64 x 7 x 7
            nn.ReLU(),
            # nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Output: 128 x 4 x 4
            # nn.ReLU(),
            # nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # Output: 256 x 2 x 2
            # nn.ReLU()
        )

        # decoder
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0), # Output: 128 x 4 x 4
            # nn.ReLU(),
            # nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=2, output_padding=1),  # Output: 64 x 7 x 7
            # nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=2, output_padding=1), # Output: 32 x 13 x 13
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=2, output_padding=1), # Output: 16 x 25 x 25
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1, output_padding=0),  # Output: 3 x 50 x 50
            nn.Sigmoid()  # use Sigmoid function because the input images are normalized between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# instantiate the model
AE_before_train = AutoEncoder()

# show the model architecture
print(AE_before_train)

# make a deep copy of the AE model to keep an untrained version
AE_after_train = copy.deepcopy(AE_before_train)
```
![image](https://hackmd.io/_uploads/rJok3YDoC.png)

```python
class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent):
        super(VariationalAutoEncoder, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Output: 16 x 25 x 25
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Output: 32 x 13 x 13
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Output: 64 x 7 x 7
            nn.ReLU(),
            # nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Output: 128 x 4 x 4
            # nn.ReLU(),
            # nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # Output: 256 x 2 x 2
            # nn.ReLU()
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent)
        self.fc_decode = nn.Linear(latent, 64 * 7 * 7)
        # self.fc_mu = nn.Linear(256 * 2 * 2, latent)
        # self.fc_logvar = nn.Linear(256 * 2 * 2, latent)
        # self.fc_decode = nn.Linear(latent, 256 * 2 * 2)

        # decoder
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0), # Output: 128 x 4 x 4
            # nn.ReLU(),
            # nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=2, output_padding=1),  # Output: 64 x 7 x 7
            # nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=2, output_padding=1), # Output: 32 x 13 x 13
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=2, output_padding=1), # Output: 16 x 25 x 25
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1, output_padding=0),  # Output: 3 x 50 x 50
            nn.Sigmoid()  # use Sigmoid function because the input images are normalized between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x = self.fc_decode(z)
        x = x.view(-1, 64, 7, 7)
        # x = x.view(-1, 256, 2, 2)
        x = self.decoder(x)
        return x, mu, logvar

## instantiate the model
VAE_before_train = VariationalAutoEncoder(latent = 128)

# show the model architecture
print(VAE_before_train)

# make a deep copy of the VAE model to keep an untrained version
VAE_after_train = copy.deepcopy(VAE_before_train)
```
![image](https://hackmd.io/_uploads/BJQu2tDi0.png)

### 5. Train AE & VAE models

#### The Hyperparameters Settings:
| batch size | epochs | learning rate | Gaussian noise |
|:----------:|:------:|:-------------:|:--------------:|
|     4      |   80   |     0.001     |      0.5       |

```python
def Train_AE(model, dataloader, epochs, lr):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []  # store the training loss for each epoch

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        for images, _ in dataloader:

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        train_losses.append(epoch_loss)  # store the training loss for current epoch
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

    # save AE model weights
    torch.save(model.state_dict(), 'AutoEncoder_weights.pth')

    # plot learning curve
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker='o')
    plt.title('AutoEncoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

# train the copied AE model
AE_after_train = Train_AE(AE_after_train, dataloader, epochs=80, lr=1e-3)
```

![image](https://hackmd.io/_uploads/rkxf17jjA.png)


```python
def Train_VAE(model, dataloader, epochs, lr):

    criterion = nn.BCELoss(reduction='sum')  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []  # store the training loss for each epoch

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        for images, _ in dataloader:

            optimizer.zero_grad()
            outputs, mu, logvar = model(images)
            reconstruction_loss = criterion(outputs, images)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + 0.1 * kl_divergence
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        train_losses.append(epoch_loss)  # store the training loss for current epoch
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

    # save VAE model weights
    torch.save(model.state_dict(), 'VariationalAutoEncoder_weights.pth')

    # plot learning curve
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker='o')
    plt.title('Variational AutoEncoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

## train the copied VAE model
VAE_after_train = Train_VAE(VAE_after_train, dataloader, epochs=80, lr=1e-3)
```

![image](https://hackmd.io/_uploads/rkVjW7jiA.png)


### 6. Evaluate average PSNR and SSIM
```python
## load the weights back into the models
AE_trained = AutoEncoder()  # recreate AE model architecture
AE_trained.load_state_dict(torch.load('AutoEncoder_weights.pth'))
VAE_trained = VariationalAutoEncoder(latent=128)  # recreate VAE model architecture
VAE_trained.load_state_dict(torch.load('VariationalAutoEncoder_weights.pth'))

# load the dataset
dataset_o = Eye_Dataset(data_path='./eye_dataset/eye/data.npy', label_path='./eye_dataset/eye/label.npy', transform=None)  # ensure the dataset loads original images
dataloader_o = DataLoader(dataset_o, batch_size=1, shuffle=False)
```

```python
def Save(image_array, path):
    image = Image.fromarray(image_array)
    image.save(path)
```

```python
def Evaluate(model, dataloader, image_ids, save_dir):
    model.eval()

    # create directory for reconstructed and original images
    os.makedirs(save_dir, exist_ok=True)
    original_dir = os.path.join(save_dir, "original")
    reconstructed_dir = os.path.join(save_dir, "reconstructed")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(reconstructed_dir, exist_ok=True)

    image_id = 1  # start from 1 as the image index range is 1-1476
    saved_images = 0

    PSNR_values = []
    SSIM_values = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            img, _ = data
            img = img.to(torch.float32)
            img = img.permute(0, 3, 1, 2)

            output = model(img)
            if isinstance(output, tuple):
                reconstructed_img = output[0]
            else:
                reconstructed_img = output

            # convert to numpy arrays for processing
            reconstructed_img = reconstructed_img.detach().cpu().numpy()
            original_img = img.cpu().numpy()

            # remove batch dimension and transpose to (H, W, C)
            reconstructed_img = reconstructed_img.squeeze(0).transpose(1, 2, 0)
            original_img = original_img.squeeze(0).transpose(1, 2, 0)

            # ensure images are in the correct range for saving
            reconstructed_img = np.clip(reconstructed_img, 0, 1) * 255.0
            original_img = np.clip(original_img, 0, 1) * 255.0

            # convert to uint8 for saving
            reconstructed_img = reconstructed_img.astype(np.uint8)
            original_img = original_img.astype(np.uint8)

            # compute PSNR and SSIM for each image
            PSNR = peak_signal_noise_ratio(original_img, reconstructed_img, data_range=255)
            SSIM = structural_similarity(original_img, reconstructed_img, multichannel=True, data_range=255, win_size=11, channel_axis=-1)

            PSNR_values.append(PSNR)
            SSIM_values.append(SSIM)

            # check if the current image ID is in the specified list
            if image_id in image_ids:
                # save original and reconstructed images
                Save(original_img, os.path.join(original_dir, f"original_{image_id}.png"))
                Save(reconstructed_img, os.path.join(reconstructed_dir, f"reconstructed_{image_id}.png"))
                saved_images += 1

            if saved_images >= 20:  # stop once 20 images have been saved
                break

            image_id += 1

    # calculate average PSNR and average SSIM
    avg_PSNR = np.mean(PSNR_values)
    avg_SSIM = np.mean(SSIM_values)

    return avg_PSNR, avg_SSIM
```

### 7. Save 20 reconstruction results
```python
## define image IDs and save directories for AE and VAE
image_ids = [91, 92, 93, 94, 95, 481, 482, 483, 484, 485, 936, 937, 938, 939, 940, 1446, 1447, 1448, 1449, 1450]
save_dir_AE = 'AE_reconstructions'
save_dir_VAE = 'VAE_reconstructions'

AE_PSNR, AE_SSIM = Evaluate(AE_trained, dataloader_o, image_ids, save_dir_AE)
print(f'AE - Average PSNR: {AE_PSNR}, Average SSIM: {AE_SSIM}')

VAE_PSNR, VAE_SSIM = Evaluate(VAE_trained, dataloader_o, image_ids, save_dir_VAE)
print(f'VAE - Average PSNR: {VAE_PSNR}, Average SSIM: {VAE_SSIM}')
```

![image](https://hackmd.io/_uploads/r1fT1ihoC.png)


### 8. Add Gaussian noise into the trained AE & VAE respectively
```python
class AutoEncoder_Noise(nn.Module):
    def __init__(self, original_autoencoder, noise_std):
        super(AutoEncoder_Noise, self).__init__()
        self.encoder = original_autoencoder.encoder
        self.decoder = original_autoencoder.decoder
        self.noise_std = noise_std

    def forward(self, x):
        # encode the input to get the latent vector
        latent_vector = self.encoder(x)

        # add Gaussian noise to the latent vector
        noisy_latent_vector = latent_vector + torch.randn_like(latent_vector) * self.noise_std

        # decode the noisy latent vector to get the output
        output = self.decoder(noisy_latent_vector)
        return output
```

```python
def Train_AE_Noise(model, dataloader, epochs, lr):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []  # store the training loss for each epoch

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        for images, _ in dataloader:

            optimizer.zero_grad()
            outputs = model(images)  # use clean images here
            loss = criterion(outputs, images)  # compare reconstructions to original images
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        train_losses.append(epoch_loss)  # store the training loss for current epoch
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

    # save the trained model weights
    torch.save(model.state_dict(), 'AutoEncoder_Noise_weights.pth')

    # plot learning curve
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker='o')
    plt.title('AutoEncoder Training Loss with Gaussian Noise in Latent Vector')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

AE_noise = AutoEncoder_Noise(AE_before_train, noise_std=0.5)
Train_AE_Noise(AE_noise, dataloader, epochs=80, lr=1e-3)
```

![image](https://hackmd.io/_uploads/ry6S8QooC.png)


```python
class VariationalAutoEncoder_Noise(nn.Module):
    def __init__(self, original_variationalautoencoder, noise_std):
        super(VariationalAutoEncoder_Noise, self).__init__()
        self.encoder = original_variationalautoencoder.encoder
        self.fc_mu = original_variationalautoencoder.fc_mu
        self.fc_logvar = original_variationalautoencoder.fc_logvar
        self.fc_decode = original_variationalautoencoder.fc_decode
        self.decoder = original_variationalautoencoder.decoder
        self.noise_std = noise_std

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        # add Gaussian noise to the latent vector
        noisy_z = z + torch.randn_like(z) * self.noise_std

        x = self.fc_decode(noisy_z)
        x = x.view(-1, 64, 7, 7)
        x = self.decoder(x)
        return x, mu, logvar
```

```python
def Train_VAE_Noise(model, dataloader, epochs, lr):

    criterion = nn.BCELoss(reduction='sum')  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []  # store the training loss for each epoch

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        for images, _ in dataloader:

            optimizer.zero_grad()
            outputs, mu, logvar = model(images)
            reconstruction_loss = criterion(outputs, images)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + 0.1 * kl_divergence
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        train_losses.append(epoch_loss)  # store the training loss for current epoch
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

    # save VAE model weights
    torch.save(model.state_dict(), 'VariationalAutoEncoder_Noise_weights.pth')

    # plot learning curve
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker='o')
    plt.title('Variational AutoEncoder Training Loss with Gaussian Noise in Latent Vector')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

VAE_noise = VariationalAutoEncoder_Noise(VAE_before_train, noise_std=0.5)
Train_VAE_Noise(VAE_noise, dataloader, epochs=80, lr=1e-3)
```

![image](https://hackmd.io/_uploads/rk6RqQos0.png)


### 9. Save 20 reconstruction results
```python
## load the weights back into the models
AE_trained_noise = AutoEncoder_Noise(AE_before_train, noise_std=0.5)  # recreate AE model architecture
AE_trained_noise.load_state_dict(torch.load('AutoEncoder_Noise_weights.pth'))
VAE_trained_noise = VariationalAutoEncoder_Noise(VAE_before_train, noise_std=0.5)  # recreate VAE model architecture
VAE_trained_noise.load_state_dict(torch.load('VariationalAutoEncoder_Noise_weights.pth'))

# define image IDs and save directories for AE and VAE
image_ids = [91, 92, 93, 94, 95, 481, 482, 483, 484, 485, 936, 937, 938, 939, 940, 1446, 1447, 1448, 1449, 1450]
save_dir_AE = 'AE_reconstructions_noise'
save_dir_VAE = 'VAE_reconstructions_noise'

AE_PSNR, AE_SSIM = Evaluate(AE_trained_noise, dataloader_o, image_ids, save_dir_AE)
print(f'AE with Gaussian noise - Average PSNR: {AE_PSNR}, Average SSIM: {AE_SSIM}')
VAE_PSNR, VAE_SSIM = Evaluate(VAE_trained_noise, dataloader_o, image_ids, save_dir_VAE)
print(f'VAE with Gaussian noise - Average PSNR: {VAE_PSNR}, Average SSIM: {VAE_SSIM}')
```

![image](https://hackmd.io/_uploads/SJS5vT3o0.png)


### 10. Experiments: Two different loss functions were introduced to preserve high-frequency information

###### I utilized the following two loss functions to replace the previous Train_VAE function in order to compare the reconstruction effects of different loss functions.

#### Total Variation Loss Experiment
```python

import torch.nn.functional as F

def total_variation_loss(image):
    """Calculates the Total Variation Loss for a given image"""
    loss = torch.sum(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])) + \
           torch.sum(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]))
    return loss

def Train_VAE(model, dataloader, epochs, lr, lambda_tv):

    criterion = nn.BCELoss(reduction='sum')  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []  # store the training loss for each epoch

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        for images, _ in dataloader:

            optimizer.zero_grad()
            outputs, mu, logvar = model(images)

            # Reconstruction Loss
            reconstruction_loss = criterion(outputs, images)

            # KL Divergence
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Total Variation Loss
            tv_loss = total_variation_loss(outputs)

            # Total Loss
            loss = reconstruction_loss + 0.1 * kl_divergence + lambda_tv * tv_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        train_losses.append(epoch_loss)  # store the training loss for the current epoch
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

    # save VAE model weights
    torch.save(model.state_dict(), 'VariationalAutoEncoder_weights_TV.pth')

    # plot learning curve
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker='o')
    plt.title('Variational AutoEncoder Training Loss with Total Variation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

# train the copied VAE model
VAE_after_train = Train_VAE(VAE_after_train, dataloader, epochs=80, lr=1e-3, lambda_tv=0.1)

# load the weights back into the models
VAE_trained = VariationalAutoEncoder(latent=128)  # recreate VAE model architecture
VAE_trained.load_state_dict(torch.load('VariationalAutoEncoder_weights_TV.pth'))

# load the dataset
dataset_o = Eye_Dataset(data_path='./eye_dataset/eye/data.npy', label_path='./eye_dataset/eye/label.npy', transform=None)  # ensure the dataset loads original images
dataloader_o = DataLoader(dataset_o, batch_size=1, shuffle=False)

# define image IDs and save directories for VAE
image_ids = [91, 92, 93, 94, 95, 481, 482, 483, 484, 485, 936, 937, 938, 939, 940, 1446, 1447, 1448, 1449, 1450]
save_dir_VAE = 'VAE_reconstructions_TV'

VAE_PSNR, VAE_SSIM = Evaluate(VAE_trained, dataloader_o, image_ids, save_dir_VAE)
print(f'VAE with TV - Average PSNR: {VAE_PSNR}, Average SSIM: {VAE_SSIM}')
```

#### Frequency-Domain Loss Experiment
```python

import torch.nn.functional as F
import torch.fft

def frequency_domain_loss(outputs, targets):
    # Compute the Fourier Transform of the output and target images
    outputs_fft = torch.fft.fftn(outputs, dim=(-2, -1))
    targets_fft = torch.fft.fftn(targets, dim=(-2, -1))

    # Calculate the magnitude (or absolute value) to get real values
    outputs_magnitude = torch.abs(outputs_fft)
    targets_magnitude = torch.abs(targets_fft)

    # Define the loss as the L2 difference between the magnitudes
    loss = F.mse_loss(outputs_magnitude, targets_magnitude)
    return loss

def Train_VAE(model, dataloader, epochs, lr, lambda_fd=0.1):
    criterion = nn.BCELoss(reduction='sum')  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []  # Store the training loss for each epoch

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, _ in dataloader:
            optimizer.zero_grad()
            outputs, mu, logvar = model(images)

            # Compute the standard reconstruction and KL divergence loss
            reconstruction_loss = criterion(outputs, images)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Compute the frequency-domain loss
            fd_loss = frequency_domain_loss(outputs, images)

            # Combine the losses with lambda_fd weighting for the frequency-domain loss
            loss = reconstruction_loss + 0.1 * kl_divergence + lambda_fd * fd_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        train_losses.append(epoch_loss)  # Store the training loss for current epoch
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

    # Save VAE model weights
    torch.save(model.state_dict(), 'VariationalAutoEncoder_weights_FD.pth')

    # Plot learning curve
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker='o')
    plt.title('Variational AutoEncoder Training Loss with Frequency-Domain Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

# Train the VAE model with the frequency-domain loss
VAE_after_train = Train_VAE(VAE_after_train, dataloader, epochs=80, lr=1e-3, lambda_fd=0.5)

# load the weights back into the models
VAE_trained = VariationalAutoEncoder(latent=128)  # recreate VAE model architecture
VAE_trained.load_state_dict(torch.load('VariationalAutoEncoder_weights_FD.pth'))

# load the dataset
dataset_o = Eye_Dataset(data_path='./eye_dataset/eye/data.npy', label_path='./eye_dataset/eye/label.npy', transform=None)  # ensure the dataset loads original images
dataloader_o = DataLoader(dataset_o, batch_size=1, shuffle=False)

# define image IDs and save directories for VAE
image_ids = [91, 92, 93, 94, 95, 481, 482, 483, 484, 485, 936, 937, 938, 939, 940, 1446, 1447, 1448, 1449, 1450]
save_dir_VAE = 'VAE_reconstructions_FD'

VAE_PSNR, VAE_SSIM = Evaluate(VAE_trained, dataloader_o, image_ids, save_dir_VAE)
print(f'VAE with FD - Average PSNR: {VAE_PSNR}, Average SSIM: {VAE_SSIM}')
```