import cv2
import numpy as np
import torch
import os
import glob
from PIL import Image
from transformers import pipeline
from diffusers import (
    ControlNetModel,
    MultiControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler
)
from diffusers.utils import load_image

"""
執行指令: 
        CUDA_VISIBLE_DEVICES=8 python inference.py
"""

# ==========================================
# 1. 設定路徑與參數
# ==========================================
# --- 模型參數 ---
epoch = 100
base_path = "./checkpoints"
checkpoint_depth_path = f"{base_path}/checkpoint-{epoch}"    # ControlNet-Depth 權重路徑
checkpoint_tile_path = f"{base_path}/checkpoint-{epoch}_1"   # ControlNet-Tile 權重路徑
base_model_path = "runwayml/stable-diffusion-v1-5"

# 指定 GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 處理資料夾參數 (請修改這裡) ---
INPUT_FOLDER = "./dataset_split/test/input_images"  # 📌 待處理的圖片資料夾路徑
OUTPUT_FOLDER = "./predictions" # 📌 結果儲存資料夾

# ==========================================
# 3. 準備 Depth 預處理器 (優化：使用指定的 DEVICE)
# ==========================================
print("Loading Depth Estimator...")
try:
    # 使用 DEVICE 變數來指定載入設備
    depth_estimator = pipeline(
        'depth-estimation', 
        model='Intel/dpt-large', 
        device=DEVICE.index if DEVICE.type == 'cuda' else -1 # 傳遞 GPU 索引或 -1 給 CPU
    )
except NameError:
    # 如果深度估計器已經載入，則跳過
    pass

def get_depth_map(image):
    depth_image = depth_estimator(image)['depth']
    depth_image = np.array(depth_image)
    depth_image = depth_image[:, :, None]
    depth_image = np.concatenate([depth_image, depth_image, depth_image], axis=2)
    return Image.fromarray(depth_image)

# ==========================================
# 4. 載入模型 (已正確使用 DEVICE)
# ==========================================
print("Loading ControlNets...")
cnet_depth = ControlNetModel.from_pretrained(checkpoint_depth_path, torch_dtype=torch.float16).to(DEVICE)
cnet_tile = ControlNetModel.from_pretrained(checkpoint_tile_path, torch_dtype=torch.float16).to(DEVICE)
multi_controlnet = MultiControlNetModel([cnet_depth, cnet_tile])

print("Loading Pipeline...")
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    base_model_path,
    controlnet=multi_controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
).to(DEVICE)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# 由於已經指定 DEVICE，通常不再需要 enable_model_cpu_offload，但保留無妨
pipe.enable_model_cpu_offload() 

# ==========================================
# 5. 批次處理函式 (核心邏輯)
# ==========================================
def process_folder(input_folder, output_folder, pipe):
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n--- Output folder created at: {output_folder} ---")

    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))
    image_paths.extend(glob.glob(os.path.join(input_folder, "*.jpeg")))
    image_paths.extend(glob.glob(os.path.join(input_folder, "*.png")))
    
    if not image_paths:
        print(f"No images found in {input_folder}. Exiting.")
        return

    print(f"Found {len(image_paths)} images to process.")

    # --- 固定 Inference 參數 ---
    prompt = "low angle view, long legs, small head, high quality, masterpiece, photorealistic"
    negative_prompt = "big head, short legs, distorted, blurry, bad anatomy, missing fingers, bad quality"
    controlnet_scales = [0.8, 0.9] # [Depth, Tile]
    strength_value = 0.7 
    
    for i, image_path in enumerate(image_paths):
        file_name = os.path.basename(image_path)
        print(f"\n[{i+1}/{len(image_paths)}] Processing: {file_name}")

        try:
            # 載入並調整大小
            original_image = load_image(image_path)
            # 保持長寬比，將最短邊縮放到 512
            if min(original_image.size) != 512:
                w, h = original_image.size
                ratio = 512 / min(w, h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                original_image = original_image.resize((new_w, new_h))
                print(f"  Resized to {new_w}x{new_h}.")
            
            # [Step A] 透視校正 (Warp)
            warped_image = original_image
            # warped_image = fix_high_angle_perspective(original_image)
            
            # [Step B] 製作 Condition 1: Depth Map 
            cond_depth = get_depth_map(warped_image)
            
            # [Step C] 製作 Condition 2: Tile
            cond_tile = warped_image

            control_images = [cond_depth, cond_tile]

            # [Step D] ControlNet Img2Img 生成
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=warped_image,
                control_image=control_images,
                num_inference_steps=30,
                guidance_scale=7.5,
                strength=strength_value, 
                controlnet_conditioning_scale=controlnet_scales,
            ).images[0]

            # [Step E] 儲存結果
            output_path = os.path.join(output_folder, f"warped_{file_name}")
            result.save(output_path)
            print(f"Result saved to {output_path}")

        except Exception as e:
            print(f"Failed to process {file_name}: {e}")
            continue

# ==========================================
# 6. 執行批次處理
# ==========================================
if __name__ == "__main__":
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER, pipe)
    print("\n--- Batch processing finished! ---")