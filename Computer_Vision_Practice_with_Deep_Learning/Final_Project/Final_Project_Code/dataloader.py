import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
# 用於使用兩個control net的，可能會需要depth map或edge map之類的
class MultiControlDataset(Dataset):
    def __init__(self, root_dir="./dataset", size=512):
        # 假設你的 dataset 資料夾結構如下：
        # ./dataset/
        #    ├── ground_truth/   (目標圖 RGB)
        #    ├── input_images/   (原始圖 RGB - 用來生 Depth 的來源)
        #    └── input_depth/    (我們會把生成的 Depth 存這裡)
        
        self.gt_dir = os.path.join(root_dir, "ground_truth")
        self.canny_dir = os.path.join(root_dir, "warp_depth") # 注意：這是從 GT 生成的 Depth
        self.tile_dir = os.path.join(root_dir, "warp_input")   # 注意：這是原始壞圖
        self.size = size
        
        # 取得檔名列表 (以 GT 資料夾為基準)
        self.filenames = [f for f in os.listdir(self.gt_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.filenames.sort()

        # 圖像轉換
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # 1. 讀取 Ground Truth (目標 RGB)
        gt_path = os.path.join(self.gt_dir, filename)
        target_image = Image.open(gt_path).convert("RGB")
        
        # 2. 讀取 Condition 1: Edge (結構)
        # 這裡讀取的是你先用 script 跑出來的 GT Depth Map
        canny_path = os.path.join(self.canny_dir, filename)
        if not os.path.exists(canny_path):
            raise FileNotFoundError(f"找不到邊緣圖: {canny_path}，請確認是否已生成 GT 的 Edge Map！")
        canny_image = Image.open(canny_path).convert("RGB")
        
        # 3. 讀取 Condition 2: Tile (細節/原始圖)
        tile_path = os.path.join(self.tile_dir, filename)
        if not os.path.exists(tile_path):
             raise FileNotFoundError(f"找不到 Input Image: {tile_path}")
        tile_image = Image.open(tile_path).convert("RGB")

        # --- 轉換 Tensor ---
        
        # Target (給 SD VAE): 正規化到 [-1, 1]
        target_tensor = self.transform(target_image) * 2.0 - 1.0
        
        # Condition Edge (給 ControlNet): 正規化到 [0, 1]
        canny_tensor = self.transform(canny_image)
        
        # Condition Tile (給 ControlNet): 正規化到 [0, 1]
        tile_tensor = self.transform(tile_image)
        
        return {
            "target": target_tensor,      # 對應 training code 的 batch["target"]
            "cond_depth": canny_tensor,   # 對應 training code 的 batch["cond_depth"]
            "cond_tile": tile_tensor      # 對應 training code 的 batch["cond_tile"]
        }