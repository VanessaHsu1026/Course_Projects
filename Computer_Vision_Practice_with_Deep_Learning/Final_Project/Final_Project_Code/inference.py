import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, MultiControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from transformers import pipeline # 用來生成深度圖
from PIL import Image
import numpy as np
import os

# 1. 設定路徑與參數
# 假設你跑了 100 epoch，路徑應該會長這樣
# 如果你的存檔邏輯跟之前一樣自動分成了兩個資料夾，請確認名稱
epoch = 60 
base_path = "./test" 
checkpoint_depth_path = f"{base_path}/checkpoint-{epoch}"    # 對應列表第一個 (Depth)
checkpoint_tile_path = f"{base_path}/checkpoint-{epoch}_1"   # 對應列表第二個 (Tile)

base_model_path = "runwayml/stable-diffusion-v1-5"
test_image_path = "./dataset/test/1.jpg" # 測試圖片路徑

# 2. 準備 Depth 預處理器 (因為我們需要把照片轉成深度圖)
# 這裡使用 transformers 內建的深度估計模型
print("Loading Depth Estimator...")
depth_estimator = pipeline('depth-estimation', model='Intel/dpt-large')

def get_depth_map(image):
    # 預測深度
    depth_image = depth_estimator(image)['depth']
    depth_image = np.array(depth_image)
    depth_image = depth_image[:, :, None]
    # ControlNet 需要 3 通道 (RGB)，所以複製 3 份
    depth_image = np.concatenate([depth_image, depth_image, depth_image], axis=2)
    return Image.fromarray(depth_image)

# 3. 載入訓練好的 MultiControlNet
print("Loading ControlNets...")
# 你的訓練順序是 [cnet_depth, cnet_tile]
# 所以第一個載入 Depth，第二個載入 Tile

cnet_depth = ControlNetModel.from_pretrained(
    checkpoint_depth_path,
    torch_dtype=torch.float16
).to("cuda")

cnet_tile = ControlNetModel.from_pretrained(
    checkpoint_tile_path,
    torch_dtype=torch.float16
).to("cuda")

multi_controlnet = MultiControlNetModel([cnet_depth, cnet_tile])

# 4. 建立 Pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=multi_controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# 5. 準備控制圖
original_image = load_image(test_image_path)
# 建議 resize 成 512x512 或 512x768 避免顯存不足或人體變形
original_image = original_image.resize((512, 512))

print("Processing images...")
# 製作 Condition 1: Depth Map (這會花一點點時間)
cond_depth = get_depth_map(original_image)

# 製作 Condition 2: Tile (直接用原圖)
cond_tile = original_image

# 6. 開始生成
# 順序必須嚴格對應：[Depth圖, 原圖]
control_images = [cond_depth, cond_tile]

# 權重設定
# 通常 Depth 給 1.0 維持結構，Tile 給 1.0 維持細節
# 如果覺得臉不像，可以稍微降低 Depth (例如 0.8) 讓 SD 多發揮一點
controlnet_scales = [1.0, 1.0] 

prompt = "a photo of a person, high quality, masterpiece, photorealistic"
negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

print("Generating...")
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=control_images,
    num_inference_steps=30,
    guidance_scale=7.5,
    controlnet_conditioning_scale=controlnet_scales,
).images[0]

# 7. 儲存結果
save_filename = f"./result.png"
result.save(save_filename)

# 順便把深度圖存下來看看對不對
cond_depth.save(f"./depth_map.png")

print(f"Done! Result saved to {save_filename}")