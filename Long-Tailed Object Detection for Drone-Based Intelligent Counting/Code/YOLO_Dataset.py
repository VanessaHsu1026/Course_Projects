import os
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image  # 匯入 Pillow 庫

# --- 設定 ---
SOURCE_TRAIN_DIR = Path('train') # 原始的 'train' 資料夾
SPLIT_DIR = Path('splits')       # 'split.py' 產生的 txt 檔案所在資料夾
DATASET_ROOT = Path('YOLO_Dataset') # 準備給 YOLO 訓練的新資料夾

# --- 函式定義 ---

def read_ids(id_file_path):
    """從 .txt 檔案讀取影像 ID 列表"""
    with open(id_file_path, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids

def convert_to_yolo_format(class_id, x, y, w, h, img_width, img_height):
    """
    將 (x, y, w, h) 絕對像素座標 (左上角)
    轉換為 YOLO 格式 (x_center_norm, y_center_norm, w_norm, h_norm)
    """
    x_center = x + w / 2
    y_center = y + h / 2
    
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n"

def process_and_copy_files(ids, split_name, source_dir, dest_root):
    """
    複製影像, 並 *轉換* 標註檔案至 YOLO 格式
    
    Args:
        ids (list): 影像 ID 列表
        split_name (str): 'train' 或 'val'
        source_dir (Path): 原始 'train' 資料夾
        dest_root (Path): YOLO 資料集的根目錄
    """
    img_dest_dir = dest_root / 'images' / split_name
    lbl_dest_dir = dest_root / 'labels' / split_name
    
    # 建立目標資料夾
    img_dest_dir.mkdir(parents=True, exist_ok=True)
    lbl_dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n正在處理並複製 {split_name} 資料 (共 {len(ids)} 筆)...")
    
    for img_id in tqdm(ids, desc=f"Processing {split_name} files"):
        img_source_path = source_dir / f"{img_id}.png"
        lbl_source_path = source_dir / f"{img_id}.txt"
        
        img_dest_path = img_dest_dir / f"{img_id}.png"
        lbl_dest_path = lbl_dest_dir / f"{img_id}.txt"
        
        # 複製影像
        if img_source_path.exists():
            shutil.copy(img_source_path, img_dest_path)
        else:
            print(f"警告: 找不到影像 {img_source_path}")
            continue # 如果影像不存在, 標註也無意義

        # 轉換標註
        if not lbl_source_path.exists():
            print(f"警告: 找不到標註 {lbl_source_path}")
            continue
            
        # 獲取影像尺寸
        try:
            with Image.open(img_source_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"錯誤: 無法讀取影像 {img_source_path} 尺寸: {e}")
            continue
            
        # 讀取原始標註, 轉換並寫入新標註
        with open(lbl_source_path, 'r') as in_f, open(lbl_dest_path, 'w') as out_f:
            lines = in_f.readlines()
            
            # 處理負樣本 (空檔案)
            if len(lines) == 0:
                # 建立一個空的 .txt 檔案, 這對負樣本訓練很重要
                continue 
                
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) != 5:
                    print(f"警告: 格式錯誤於 {lbl_source_path}: {line}")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    # 執行轉換
                    yolo_line = convert_to_yolo_format(class_id, x, y, w, h, img_width, img_height)
                    out_f.write(yolo_line)
                    
                except ValueError as e:
                    print(f"警告: 無法解析 {lbl_source_path} 中的行: {line}, 錯誤: {e}")

def main():
    print(f"YOLO 資料集根目錄將建立於: {DATASET_ROOT.resolve()}")
    
    # 讀取切割好的 ID
    train_id_file = SPLIT_DIR / 'train_ids.txt'
    val_id_file = SPLIT_DIR / 'val_ids.txt'
    
    if not train_id_file.exists() or not val_id_file.exists():
        print(f"錯誤: 找不到 {train_id_file} 或 {val_id_file}")
        print("請先執行 'python split.py' 產生切割檔案")
        return

    train_ids = read_ids(train_id_file)
    val_ids = read_ids(val_id_file)
    
    # 處理並複製檔案
    process_and_copy_files(train_ids, 'train', SOURCE_TRAIN_DIR, DATASET_ROOT)
    process_and_copy_files(val_ids, 'val', SOURCE_TRAIN_DIR, DATASET_ROOT)
    
    print("\n" + "="*50)
    print("YOLO 資料集準備完成！ (包含格式轉換)")
    print(f"請檢查 '{DATASET_ROOT}' 資料夾")
    print("下一步: 請執行 'python HW2.py'")
    print("="*50)

if __name__ == "__main__":
    main()