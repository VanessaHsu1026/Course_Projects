---
title: 'HW2: Generative Adversarial Network'

---

# HW2: Generative Adversarial Network
##### In this project, I will utilize PyTorch to implement DCGAN and LSGAN, training both models on the MNIST dataset. I will conduct both quantitative and qualitative analyses to compare their performance. Additionally, Generative Adversarial Networks (GANs) consist of two competing components: a generator and a discriminator, which collaborate to produce realistic images. I will also evaluate the Fréchet Inception Distance (FID) between the training images and the generated images, with the goal of achieving a value below 120.

| Model Type |  FID   |
|:----------:|:------:|
|   <font color="#3363ff">**DCGAN**</font>    | <font color="#ff3333">**112.98**</font> |
|   <font color="#3363ff">**LSGAN**</font>    | <font color="#ff3333">**103.03**</font> |

## Environment
###### Step 1. Download `HW2_徐綉惠.zip` from EECLASS
###### Step 2. Upload the `HW2_徐綉惠` folder to your Google Drive
###### Step 3. Open the file `HW2.ipynb`
###### Step 4. Please execute each cell sequentially from `1` to `8`, and then proceed to cells `12` and `14` 

### 1. Load Model Weights
```python
from google.colab import drive
drive.mount('/content/drive')
!ls /content/drive/
%cd "/content/drive/My Drive/HW2_徐綉惠/"
!ls "./weights/"
```
![image](https://hackmd.io/_uploads/B1HaBzMmJe.png)

### 2. Import Libraries
```python
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import torchvision.utils as vutils
from tqdm.auto import tqdm
import os

import matplotlib.pyplot as plt
%matplotlib inline
```

### 3. Set Random Seeds for Reproducibility
```python
import random

def set_seed(seed):
    torch.manual_seed(seed)  # Set seed for PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Set seed for PyTorch GPU
    np.random.seed(seed)  # Set seed for NumPy
    random.seed(seed)  # Set seed for Python's random library

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)  # Call the function to set the seed
```

### 4. Define Some Useful Functions
```python
def show_images(images, epoch = None, save_dir = None):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    plt.figure()
    if epoch != None:
        plt.title("Generator_epoch_{}".format(epoch))
    plt.axis('off')
    for index, image in enumerate(images):
        plt.subplot(sqrtn, sqrtn, index+1)
        plt.imshow(image.reshape(28, 28))
        plt.axis('off')

    if epoch != None:
        filename = "Generator_epoch_{}.png".format(epoch)
    else:
        filename = "Generator.png"

    if save_dir:
        os.makedirs(save_dir, exist_ok = True)
        plt.savefig(os.path.join(save_dir, filename))
```

### 5. Load MNIST Dataset
```python
train_set = datasets.MNIST('./dataset', train=True, download=True, transform=transforms.ToTensor())
```

### 6. Define Basic Building Blocks
```python
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Conv_BN_ReLU(nn.Sequential):       # convolution -> Batch Normalization -> ReLU
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        norm_layer = nn.BatchNorm2d
        super(Conv_BN_ReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU(inplace=True),
        )
class Conv_BN_LReLU(nn.Sequential):      # convolution -> Batch Normalization -> LeakyReLU
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        norm_layer = nn.BatchNorm2d
        super(Conv_BN_LReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.LeakyReLU(inplace=True),
        )
class TConv_BN_ReLU(nn.Sequential):      # Transposed convolution -> Batch Normalization -> ReLU
    def __init__(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1):
        padding = (kernel_size - 1) // 2
        norm_layer = nn.BatchNorm2d
        super(TConv_BN_ReLU, self).__init__(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=False),
            norm_layer(out_planes),
            nn.ReLU(inplace=True),
        )
```

### 7. The Structures of My Best Model (Generator & Discriminator)
```python
class Generator(nn.Module):
    def __init__(self, latents):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            # Input: random_Z, state size: latents x 1 x 1
            TConv_BN_ReLU(latents, 256, 4, 2, 1),   # State size: 256 x 2 x 2
            Conv_BN_ReLU(256, 128, 3, 1)
        )

        self.layer2 = nn.Sequential(
            TConv_BN_ReLU(128, 256, 4, 1, 0),        # State size: 256 x 3 x 3
            TConv_BN_ReLU(256, 256, 4, 2, 1)         # State size: 256 x 6 x 6
        )

        self.layer3 = nn.Sequential(
            TConv_BN_ReLU(256, 128, 4, 1, 0),         # State size: 256 x 7 x 7
            TConv_BN_ReLU(128, 128, 4, 2, 1),         # State size: 256 x 14 x 14
            Conv_BN_ReLU(128, 128, 3, 1)              # State size: 256 x 6 x 6
        )

        self.layer4 = nn.Sequential(
            TConv_BN_ReLU(128, 64, 4, 2, 1),         # State size: 64 x 28 x 28
            Conv_BN_ReLU(64, 64, 3, 1),
            Conv_BN_ReLU(64, 64, 3, 1),
            nn.Conv2d(64, 1, 3, 1, 1),               # Output: 1 x 28 x 28
            nn.Tanh()
        )

    def forward(self, x):                           # Forward pass of G
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            Conv_BN_LReLU(1, 32, 3, 2),          # State size: 32 x 14 x 14
            Conv_BN_LReLU(32, 64, 3, 1),         # State size: 64 x 14 x 14
            Conv_BN_LReLU(64, 128, 3, 2),        # State size: 128 x 7 x 7
            Conv_BN_LReLU(128, 128, 3, 2),       # State size: 32 x 3 x 3
            Conv_BN_LReLU(128, 64, 3, 2)         # State size: 32 x 1 x 1
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):                       # forward pass of D
        x = self.conv(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        ft = x
        output = self.fc(x)

        return output
```

### 8. Training Settings
```python
config = {
    "model_type": "LSGAN",  # Model type (options: DCGAN, LSGAN)
    "batch_size": 200,      # Batch size during training
    "lr": 0.0002,           # Learning rate for optimizers
    "epochs": 20,           # Number of training epochs
    "latent_dim": 60,       # input dimension of generator
    "ckpt_dir": './checkpoints',    # directory for saving checkpoints
    "result_dir": './results'       # directory for saving images
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on Device: {device}')

# claim generator and discriminator
G = Generator(latents = config['latent_dim']).to(device)
D = Discriminator().to(device)
G.apply(weights_init)
D.apply(weights_init)

# claim optimizer and scheduler for G and D
g_optimizer = optim.Adam(G.parameters(), lr=config['lr'], betas=(0.5, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=config['lr'], betas=(0.5, 0.999))

g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=5, gamma=0.5)
d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=5, gamma=0.5)

# change loss function based on model type
if config['model_type'] == "DCGAN":
    adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)
elif config['model_type'] == 'LSGAN':
    adversarial_loss = torch.nn.MSELoss().to(device)
else:
    print("Unknown model type")

# Create training dataloader
train_loader = DataLoader(train_set, batch_size = config['batch_size'], shuffle=True)
```

### 9. Training Procedure
##### :sparkles: If you don't want to train the model again, please skip to "load model" part.
```python
G.train()
D.train()
loss_g, loss_d = [], []
start_time= time.time()

for epoch in tqdm(range(config['epochs'])):
    total_loss_g, total_loss_d = 0, 0
    for i_iter, (images, label) in enumerate(train_loader):

        # ---------------------------------- The section you need to modified starts here ----------------------------------
        # -----------------
        #  Stage I: Training Discriminator
        # -----------------

        # step 1. create random noise for G
        noise = torch.randn(images.size(0), config['latent_dim']).view(images.size(0), config['latent_dim'], 1, 1)
        noise = noise.to(device)

        # step 2. generate fake images with G, and pass them to D
        fake_inputs = G(noise)  # Generate fake images using the generator
        fake_outputs = D(fake_inputs.detach())  # Pass fake images to the discriminator (detach to avoid backprop to G)

        # step 3. create fake labels for fake inputs
        fake_labels = torch.zeros(fake_inputs.size(0), 1, device=device)  # Fake images labeled as 0

        # step 4. calculate fake loss, e.g., cross entropy between fake_outputs and fake label
        fake_loss = adversarial_loss(fake_outputs, fake_labels)

        # step 5. pass real images to D
        real_inputs = images.to(device)  # Move real images to the same device
        real_outputs = D(real_inputs)  # Pass real images to the discriminator

        # step 6. create real labels for real inputs
        real_labels = torch.ones(real_inputs.size(0), 1, device=device)  # Real images labeled as 1

        # step 7. calculate real loss, e.g., cross entropy between real_outputs and real label
        real_loss = adversarial_loss(real_outputs, real_labels)


        # step 8. take average of real_loss and fake_loss
        loss_d_value = (real_loss + fake_loss) / 2

        # step 9. update D according to the loss
        # G is freezed in this stage, so try to prevent gradient descent of G
        d_optimizer.zero_grad()
        loss_d_value.backward()
        d_optimizer.step()

        # accumulate total loss of D
        total_loss_d += loss_d_value


        # -----------------
        #  Stage II: Training Generator
        # -----------------

        # step 1. create random noise for G
        noise = torch.randn(images.size(0), config['latent_dim']).view(images.size(0), config['latent_dim'], 1, 1)
        noise = noise.to(device)

        # step 2. To generate fake images, pass noise to G
        fake_inputs = G(noise) # Generate fake images using the generator

        # step 3. pass fake_inputs to D
        fake_outputs = D(fake_inputs) # Discriminator evaluates the fake images

        # step 4. create fake labels for fake outputs (Note: the label value should be different from the one in stage I)
        fake_labels = torch.ones(fake_inputs.size(0), 1, device=device)  # Fake images labeled as real (1.0)

        # step 5. calculate loss of G (fake loss), e.g., cross entropy between fake_outputs and fake label
        loss_g_value = adversarial_loss(fake_outputs, fake_labels)

        # step 6. update G according to the loss
        g_optimizer.zero_grad()
        loss_g_value.backward()
        g_optimizer.step()

        # accumulate total loss of G
        total_loss_g += loss_g_value

        # ---------------------------------- The section you need to modified ends here ----------------------------------



    total_loss_g /= len(train_loader)
    total_loss_d /= len(train_loader)
    loss_g.append(total_loss_g.item())
    loss_d.append(total_loss_d.item())

    g_scheduler.step()
    d_scheduler.step()

    print('[Epoch: {}/{}] D_loss: {:.4f} G_loss: {:.4f}'.format(epoch, config['epochs'],
        total_loss_d.item(), total_loss_g.item()))

    if epoch % 5 == 0 or epoch == config['epochs'] - 1:
        # save generate image and checkpoint for each 5 epochs intervals or the last epochs
        print('Generated images for epoch: {}'.format(epoch))
        imgs_numpy = fake_inputs.data.cpu().numpy()
        show_images(imgs_numpy[:16], epoch, config['result_dir'])
        os.makedirs(config['ckpt_dir'], exist_ok = True)
        # torch.save(G, os.path.join(config['ckpt_dir'], 'DCGAN_Generator_epoch{:04d}.pth'.format(epoch))) # save DCGAN model
        torch.save(G, os.path.join(config['ckpt_dir'], 'LSGAN_Generator_epoch{:04d}.pth'.format(epoch))) # save LSGAN model

print('Training Finished.')
print('Cost Time: {:.3f}s'.format(time.time() - start_time))
```

### 10. Plot Loss Curves
```python
epochs = range(1, len(loss_g) + 1)  # Generate epoch numbers (starting from 1)

# Plot the losses
plt.figure(figsize=(10, 6))  # Set figure size
plt.plot(epochs, loss_g, label="Generator Loss", marker='o')  # Plot Generator loss
plt.plot(epochs, loss_d, label="Discriminator Loss", marker='s')  # Plot Discriminator loss

# Add title and labels
plt.title("Loss Curves for Generator and Discriminator", fontsize=16)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Loss", fontsize=14)

# Add legend
plt.legend(fontsize=12)

# Add grid for better readability
plt.grid(alpha=0.5)

# Show the plot
plt.show()
```

### 11. Show Generated Images
```python
# latent_dim = 100         # latent_dim of DCGAN model
latent_dim = 60         # latent_dim of LSGAN model
num_image = 16          # number of images to be generated
device = 'cuda' if torch.cuda.is_available() else 'cpu'
G = Generator(latents = latent_dim).to(device)
print(f'Running on Device: {device}')



def inference(generator, latent_dim, num_image, device, checkpoint = None):
    '''
    Generate images given generator
    '''
    if checkpoint:
        generator = torch.load(checkpoint, map_location=device)

    torch.manual_seed(0)
    generator.eval()
    noise = torch.randn(num_image, latent_dim, 1, 1)
    noise = noise.to(device)
    gen_imgs = generator(noise)

    return gen_imgs

# load checkpoints and generate images

# imgs = inference(G, latent_dim, num_image, device, checkpoint = 'checkpoints/DCGAN_Generator_epoch0029.pth') # DCGAN generate images
imgs = inference(G, latent_dim, num_image, device, checkpoint = 'checkpoints/LSGAN_Generator_epoch0019.pth') # LSGAN generate images
imgs_numpy = imgs.data.cpu().numpy()
show_images(imgs_numpy[:16])
```

### 12. Compute Fréchet Inception Distance (FID)
```python
from torchvision.models.inception import inception_v3
from torch.nn import functional as F
from scipy.stats import entropy
from scipy import linalg

mean_inception = [0.485, 0.456, 0.406]
std_inception = [0.229, 0.224, 0.225]


def compute_FID(img1, img2, batch_size=1, resize = True):
    '''
        Compute FID give two image set, img1 and img2.
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    N1 = len(img1)
    N2 = len(img2)
    n_act = 1000  # the number of final layer's dimension

    # Set up dataloaders
    dataloader1 = torch.utils.data.DataLoader(img1, batch_size=batch_size)
    dataloader2 = torch.utils.data.DataLoader(img2, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    # get the activations
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)

    def get_activations(x, resize = resize):
        if resize:
            x = up(x)
        x = inception_model(x)[0]
        return x.cpu().data.numpy().reshape(batch_size, -1)

    act1 = np.zeros((N1, n_act))
    act2 = np.zeros((N2, n_act))

    data = [dataloader1, dataloader2]
    act = [act1, act2]
    for n, loader in enumerate(data):
        for i, batch in enumerate(loader):
            batch = batch['image'].to(device)
            batch_size_i = batch.size()[0]
            activation = get_activations(batch)

            act[n][i * batch_size:i * batch_size + batch_size_i] = activation

    # compute the activation's statistics: mean and std
    def compute_act_mean_std(act):
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma
    mu_act1, sigma_act1 = compute_act_mean_std(act1)
    mu_act2, sigma_act2 = compute_act_mean_std(act2)

    # compute FID
    def _compute_FID(mu1, mu2, sigma1, sigma2,eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        FID = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

        return FID

    FID = _compute_FID(mu_act1, mu_act2, sigma_act1, sigma_act2)
    return FID
```
```python
class ScoreDataset(Dataset):
    """Dataset structur for evaluating FID"""

    def __init__(self, data, transform=None, has_label = False):
        self.data = data
        self.transform = transform
        self.has_label = has_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.has_label:
            image, label = self.data[idx]
        else:
            image = self.data[idx]
            label = 0

        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label}


        return sample
```

### 13. Evaluate Generator with FID
```python
# latent_dim = 100                                                 # latend dim of DCGAN model
latent_dim = 60                                                 # latend dim of LSGAN model
# checkpoint_path = 'checkpoints/DCGAN_Generator_epoch0029.pth'   # checkpoint path of DCGAN
checkpoint_path = 'checkpoints/LSGAN_Generator_epoch0019.pth'   # checkpoint path of LSGAN
num_images = 1000                                               # This number should not be changed!


tf = transforms.Compose([       # transform images to fit format of InceptionV3
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
    transforms.Normalize(mean_inception, std_inception)
    ])


gen_imgs = inference(G, latent_dim, num_images, device, checkpoint = checkpoint_path)
gen_data = ScoreDataset(gen_imgs, transform = tf, has_label = False)
train_subset = [train_set[i] for i in range(num_images)]
train_data = ScoreDataset(train_subset, transform = tf, has_label = True)

FID = compute_FID(train_data, gen_data, batch_size = 1)
print('FID is {:.2f}'.format(FID))
```

### 14. Inference (My Best Model)
##### :sparkles: Please skip to here to "load model".
```python
def inference(generator, latent_dim, num_image, device, checkpoint = None):
    '''
    Generate images given generator
    '''
    if checkpoint:
        generator = torch.load(checkpoint, map_location=device)

    torch.manual_seed(0)
    generator.eval()
    noise = torch.randn(num_image, latent_dim, 1, 1)
    noise = noise.to(device)
    gen_imgs = generator(noise)

    return gen_imgs
```
#### :+1: DCGAN
```python
latent_dim = 100         # latent_dim of DCGAN model
num_image = 16          # number of images to be generated
checkpoint_path = 'weights/DCGAN_Generator_epoch0029.pth'   # checkpoint path of DCGAN
num_images = 1000                         # This number should not be changed!
G = torch.load(checkpoint_path, map_location=device)
G.to(device)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load checkpoints and generate images
imgs = inference(G, latent_dim, num_image, device, checkpoint = checkpoint_path)
imgs_numpy = imgs.data.cpu().numpy()
show_images(imgs_numpy[:16])

tf = transforms.Compose([       # transform images to fit format of InceptionV3
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
    transforms.Normalize(mean_inception, std_inception)
    ])


gen_imgs = inference(G, latent_dim, num_images, device, checkpoint = checkpoint_path)
gen_data = ScoreDataset(gen_imgs, transform = tf, has_label = False)
train_subset = [train_set[i] for i in range(num_images)]
train_data = ScoreDataset(train_subset, transform = tf, has_label = True)

FID = compute_FID(train_data, gen_data, batch_size = 1)
print('FID is {:.2f}'.format(FID))
```
![image](https://hackmd.io/_uploads/rk2kyQfXkl.png)

#### :+1: LSGAN
```python
latent_dim = 60         # latent_dim of LSGAN model
num_image = 16          # number of images to be generated
checkpoint_path = 'weights/LSGAN_Generator_epoch0019.pth'   # checkpoint path of LSGAN
num_images = 1000                         # This number should not be changed!
G = torch.load(checkpoint_path, map_location=device)
G.to(device)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load checkpoints and generate images
imgs = inference(G, latent_dim, num_image, device, checkpoint = checkpoint_path)
imgs_numpy = imgs.data.cpu().numpy()
show_images(imgs_numpy[:16])

tf = transforms.Compose([       # transform images to fit format of InceptionV3
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
    transforms.Normalize(mean_inception, std_inception)
    ])


gen_imgs = inference(G, latent_dim, num_images, device, checkpoint = checkpoint_path)
gen_data = ScoreDataset(gen_imgs, transform = tf, has_label = False)
train_subset = [train_set[i] for i in range(num_images)]
train_data = ScoreDataset(train_subset, transform = tf, has_label = True)

FID = compute_FID(train_data, gen_data, batch_size = 1)
print('FID is {:.2f}'.format(FID))
```
![image](https://hackmd.io/_uploads/rJsI1mGQJx.png)
