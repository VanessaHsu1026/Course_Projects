import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer, pipeline
from torch.utils.data import Dataset, DataLoader
from diffusers.models import MultiControlNetModel
from dataloader import MultiControlDataset
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt
import os
import lpips
from PIL import Image
import torchvision.transforms.functional as TF
import random
from tqdm import tqdm

"""
執行指令: 
        CUDA_VISIBLE_DEVICES=8 python train.py
"""

test_name = "checkpoints" # 可以自己取
os.makedirs(test_name, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnet_depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth").to(device)
cnet_tile = ControlNetModel.from_pretrained("lllyasviel/control_v11f1e_sd15_tile").to(device)
multi_controlnet = MultiControlNetModel([cnet_depth, cnet_tile])
multi_controlnet.train()

cnet_depth.requires_grad_(True)
cnet_tile.requires_grad_(True)

# 載入 SD1.5 的 VAE 和 Text Encoder (這兩個通常不訓練)
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder").to(device)
scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(device)
# 我們從一個空的 ControlNet 開始，或者從現有的載入
# controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth").to(device) # 可以讀取RGB

# lora_config = LoraConfig(
#     r=16,               # Rank: 數字越大越聰明但參數量越多 (8, 16, 32)
#     lora_alpha=32,      # Alpha: 縮放係數，通常是 Rank 的兩倍
#     target_modules=["to_q", "to_k", "to_v", "to_out.0"], # 指定要訓練 Attention 層的 Linear
#     lora_dropout=0.1,
#     bias="none"
# )

# controlnet = get_peft_model(controlnet, lora_config) # 這個peft就是可以掛lora的套件
# for name, param in controlnet.named_parameters():
#     if "lora" not in name:  
#         param.requires_grad = False # 只去更新有lora部分的參數

# controlnet.requires_grad_(True)

# controlnet.train()
# 全都freeze住
unet.requires_grad_(False)
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

train_dataset = MultiControlDataset(root_dir="./dataset_split/train") 
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
optimizer = torch.optim.AdamW(multi_controlnet.parameters(), lr=5e-5)

# prior loss
# pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
# ).to(device)
# pipe.unet.eval()
# pipe.vae.eval()
# pipe.text_encoder.eval()
if not os.path.exists("./prior_images"): # 避免每次重跑都重新生成
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None).to(device)
    pipe.set_progress_bar_config(disable=True)
    
    prior_dir = "./prior_images"
    os.makedirs(prior_dir, exist_ok=True)

    prompt = "a photo of a person" 
    num_prior = 30
    print("Generating prior images...")
    for i in range(num_prior):
        img = pipe(prompt, num_inference_steps=20).images[0]
        img.save(f"{prior_dir}/{i}.png")
    
    del pipe # 刪除 pipe 釋放顯卡記憶體，不然等下訓練會 OOM
    torch.cuda.empty_cache()

prior_dir = "./prior_images"
prior_images_list = [os.path.join(prior_dir, f) for f in os.listdir(prior_dir)]
prior_images_tensors = [TF.to_tensor(Image.open(p).convert("RGB")).unsqueeze(0).to(device) for p in prior_images_list]

# 3. 設定訓練參數
num_epochs = 100
epoch_loss_history = []
lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

def train_step(batch):
    batch_size = batch["target"].shape[0]
    # 準備 Text Embeddings (空的 Prompt)
    # text_inputs = tokenizer([""] * batch_size, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    # encoder_hidden_states = text_encoder(text_inputs.input_ids.to(device))[0]

    encoder_hidden_states = text_encoder(
        tokenizer("a photo of a person", return_tensors="pt").input_ids.to(device)
    )[0].repeat(batch_size, 1, 1)

    # B. 處理 Target Image (GT) -> 轉成 Latents -> 加雜訊
    latents = vae.encode(batch["target"].to(device)).latent_dist.sample()
    latents = latents * 0.18215

    # 加噪 (Forward Diffusion)
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, scheduler.num_train_timesteps - 10, ()).to(device)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    # C. 處理 Input Image (Condition)
    # 這是關鍵！把你的「歪頭照」當作 ControlNet 的輸入
    cond_depth = batch["cond_depth"].to(device)
    cond_tile = batch["cond_tile"].to(device)
    controlnet_images = [cond_depth, cond_tile]
    # D. 模型預測 (Forward Pass)
    # 1. ControlNet 算出控制特徵
    down_block_res_samples, mid_block_res_sample = multi_controlnet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_cond=controlnet_images, # 傳入 List
        conditioning_scale=[1.0, 1.0],
        return_dict=False,
    )

    # 2. UNet 預測雜訊
    noise_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
    ).sample

    if scheduler.config.prediction_type == "epsilon":
        target = noise
    elif scheduler.config.prediction_type == "v_prediction":
        target = scheduler.get_velocity(latents, noise, timesteps)

    # E. 計算 Loss (監督式學習)
    noise_loss = torch.nn.functional.mse_loss(noise_pred, target)
    # pred_latents = scheduler.step(noise_pred, timesteps, noisy_latents).pred_original_sample
    # pred_images = vae.decode(pred_latents / 0.18215).sample
    # pred_images = (pred_images + 1) / 2

    #recon_loss = torch.nn.functional.l1_loss(pred_images, batch["target_image"].to(device))

    # ---- LPIPS loss ----
    # gt_images = (batch["target_image"].to(device) + 1) / 2
    # lpips_loss = lpips_loss_fn(pred_images, gt_images).mean()

    # -----------------------------------------------------
    #   PRIOR PRESERVATION LOSS (DreamBooth / ID-Booth)
    # -----------------------------------------------------
    # Randomly select one prior image per batch
    prior_img = random.choice(prior_images_tensors)
    prior_latent = vae.encode(prior_img).latent_dist.sample() * 0.18215
    prior_noise = torch.randn_like(prior_latent)
    prior_t = torch.randint(0, scheduler.num_train_timesteps - 10, ()).to(device)
    noisy_prior = scheduler.add_noise(prior_latent, prior_noise, prior_t)

    # Text conditioning uses SAME prompt as training
    # prior_embeds = encoder_hidden_states[:1]   # reuse "a person standing"

    # No ControlNet for prior — follows DreamBooth
    prior_pred = unet(
        noisy_prior,
        prior_t,
        encoder_hidden_states=encoder_hidden_states[:1], # 使用相同的 embedding
    ).sample

    prior_loss = torch.nn.functional.mse_loss(prior_pred, prior_noise)


    # Combine losses (tunable weights)
    #loss = noise_loss + 0.5 * lpips_loss
    loss = noise_loss
    
    # F. 反向傳播
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()

for epoch in range(num_epochs):
    epoch_loss_sum = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in progress_bar: 
        loss_value = train_step(batch)
        epoch_loss_sum += loss_value
        progress_bar.set_postfix({"Loss": f"{loss_value:.4f}"})

    avg_epoch_loss = epoch_loss_sum / len(train_dataloader)
    epoch_loss_history.append(avg_epoch_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] Finished. Avg Loss: {avg_epoch_loss:.4f}")
            
# 計算 Epoch 平均 Loss
    plt.figure(figsize=(10, 5))
    epochs_x = range(1, len(epoch_loss_history) + 1)
    plt.plot(epochs_x, epoch_loss_history, label='Avg Epoch Loss', color='red', marker='o')
    plt.title('ControlNet Training Loss')
    plt.grid(True)
    plt.savefig(f'./{test_name}/loss_curve.png')
    plt.close()

    if (epoch + 1) % 10 == 0:
        save_path = f"./{test_name}/checkpoint-{epoch+1}"
        # 儲存 MultiControlNet
        multi_controlnet.save_pretrained(save_path)
        print(f"Saved to {save_path}")

print("Training Complete.")