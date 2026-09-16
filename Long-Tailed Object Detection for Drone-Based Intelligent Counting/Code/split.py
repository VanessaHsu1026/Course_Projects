import os
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

class DroneDatasetLoader:
    """無人機物體偵測資料集載入器"""
    
    def __init__(self, train_dir='train', test_dir='test'):
        """
        Args:
            train_dir: 訓練資料夾路徑
            test_dir: 測試資料夾路徑
        """
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.class_names = ['car', 'hov', 'person', 'motorcycle']
        
    def load_annotation(self, txt_path):
        """
        載入單一標註檔案
        
        Args:
            txt_path: 標註檔案路徑 (e.g., 'train/img0001.txt')
            
        Returns:
            list: [{'class': 0, 'bbox': [x, y, w, h]}, ...]
        """
        annotations = []
        
        if not os.path.exists(txt_path):
            print(f"警告: 找不到標註檔案 {txt_path}")
            return annotations
        
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:  # 跳過空行
                    continue
                
                # 格式: <class>,<x>,<y>,<w>,<h>
                parts = line.split(',')
                if len(parts) != 5:
                    print(f"警告: 格式錯誤的標註行: {line}")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    annotations.append({
                        'class': class_id,
                        'bbox': [x, y, w, h]
                    })
                except ValueError as e:
                    print(f"警告: 無法解析標註: {line}, 錯誤: {e}")
                    continue
        
        return annotations
    
    def load_all_train_annotations(self):
        """
        載入所有訓練集的標註
        
        Returns:
            dict: {image_id: [annotations]}
                  例如: {'img0001': [{...}, {...}], 'img0002': [...]}
        """
        all_annotations = {}
        
        # 尋找所有 txt 檔案
        txt_files = sorted(self.train_dir.glob('*.txt'))
        
        print(f"找到 {len(txt_files)} 個標註檔案")
        
        for txt_file in txt_files:
            # 取得影像 ID (不含副檔名)
            image_id = txt_file.stem  # 'img0001.txt' -> 'img0001'
            
            # 檢查對應的影像是否存在
            img_path = self.train_dir / f"{image_id}.png"
            if not img_path.exists():
                print(f"警告: 找不到對應的影像 {img_path}")
                continue
            
            # 載入標註
            annotations = self.load_annotation(txt_file)
            all_annotations[image_id] = annotations
        
        print(f"成功載入 {len(all_annotations)} 張影像的標註")
        return all_annotations
    
    def get_dataset_statistics(self, annotations):
        """
        統計資料集的類別分布
        
        Args:
            annotations: dict, {image_id: [annotations]}
            
        Returns:
            dict: 統計資訊
        """
        total_objects = 0
        class_counts = {i: 0 for i in range(4)}
        images_per_class = {i: 0 for i in range(4)}
        objects_per_image = []
        
        for image_id, anns in annotations.items():
            n_objects = len(anns)
            objects_per_image.append(n_objects)
            total_objects += n_objects
            
            # 統計每個類別
            classes_in_image = set()
            for ann in anns:
                class_id = ann['class']
                class_counts[class_id] += 1
                classes_in_image.add(class_id)
            
            # 統計包含該類別的影像數量
            for class_id in classes_in_image:
                images_per_class[class_id] += 1
        
        stats = {
            'total_images': len(annotations),
            'total_objects': total_objects,
            'class_counts': class_counts,
            'images_per_class': images_per_class,
            'avg_objects_per_image': total_objects / len(annotations) if annotations else 0,
            'objects_per_image': objects_per_image
        }
        
        return stats
    
    def print_statistics(self, stats):
        """列印資料集統計資訊"""
        print("\n" + "="*60)
        print("資料集統計")
        print("="*60)
        print(f"總影像數: {stats['total_images']}")
        print(f"總物體數: {stats['total_objects']}")
        print(f"平均每張影像物體數: {stats['avg_objects_per_image']:.2f}")
        print("\n類別分布 (物體數量):")
        for class_id, count in stats['class_counts'].items():
            class_name = self.class_names[class_id]
            percentage = (count / stats['total_objects'] * 100) if stats['total_objects'] > 0 else 0
            print(f"  {class_id}: {class_name:12s} - {count:5d} 個 ({percentage:5.2f}%)")
        
        print("\n類別分布 (影像數量):")
        for class_id, count in stats['images_per_class'].items():
            class_name = self.class_names[class_id]
            percentage = (count / stats['total_images'] * 100) if stats['total_images'] > 0 else 0
            print(f"  {class_id}: {class_name:12s} - {count:5d} 張影像 ({percentage:5.2f}%)")
        print("="*60 + "\n")
    
    def get_dominant_class_per_image(self, annotations):
        """
        為每張影像找出主要類別（用於 stratified split）
        
        Args:
            annotations: dict, {image_id: [annotations]}
            
        Returns:
            list: image_ids
            list: dominant_classes
        """
        image_ids = []
        dominant_classes = []
        
        for image_id in sorted(annotations.keys()):
            anns = annotations[image_id]
            
            if len(anns) == 0:
                # 如果沒有物體，標記為 -1 (背景類別)
                print(f"提示: {image_id} 沒有標註物體, 標記為背景類別 -1")
                dominant_class = -1
            else:
                # 統計該影像中各類別的數量
                class_counts = Counter([ann['class'] for ann in anns])
                
                # 找出出現最多的類別
                dominant_class = class_counts.most_common(1)[0][0]
            
            image_ids.append(image_id)
            dominant_classes.append(dominant_class)
        
        return image_ids, dominant_classes
    
    def stratified_split(self, annotations, test_size=0.1, random_state=42):
        """
        執行 Stratified Split
        
        Args:
            annotations: dict, {image_id: [annotations]}
            test_size: 驗證集比例 (0.1 = 10%)
            random_state: 隨機種子
            
        Returns:
            train_ids: list, 訓練集影像 ID
            val_ids: list, 驗證集影像 ID
        """
        # 取得每張影像的主要類別
        image_ids, dominant_classes = self.get_dominant_class_per_image(annotations)
        
        print(f"\n可用於切割的影像數: {len(image_ids)}")
        print("主要類別分布:")
        class_dist = Counter(dominant_classes)
        for class_id, count in sorted(class_dist.items()):
            print(f"  {self.class_names[class_id]:12s}: {count} 張")
        
        # 執行 Stratified Split
        train_ids, val_ids = train_test_split(
            image_ids,
            test_size=test_size,
            stratify=dominant_classes,
            random_state=random_state
        )
        
        # 驗證分布
        print(f"\n切割結果:")
        print(f"  訓練集: {len(train_ids)} 張 ({len(train_ids)/len(image_ids)*100:.1f}%)")
        print(f"  驗證集: {len(val_ids)} 張 ({len(val_ids)/len(image_ids)*100:.1f}%)")
        
        # 顯示訓練集類別分布
        train_classes = [dominant_classes[image_ids.index(id)] for id in train_ids]
        train_dist = Counter(train_classes)
        print("\n訓練集類別分布:")
        for class_id, count in sorted(train_dist.items()):
            print(f"  {self.class_names[class_id]:12s}: {count} 張")
        
        # 顯示驗證集類別分布
        val_classes = [dominant_classes[image_ids.index(id)] for id in val_ids]
        val_dist = Counter(val_classes)
        print("\n驗證集類別分布:")
        for class_id, count in sorted(val_dist.items()):
            print(f"  {self.class_names[class_id]:12s}: {count} 張")
        
        return train_ids, val_ids
    
    def save_split_to_file(self, train_ids, val_ids, output_dir='splits'):
        """
        將切割結果儲存到檔案
        
        Args:
            train_ids: 訓練集影像 ID
            val_ids: 驗證集影像 ID
            output_dir: 輸出資料夾
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 儲存訓練集 ID
        train_file = output_path / 'train_ids.txt'
        with open(train_file, 'w') as f:
            for img_id in train_ids:
                f.write(f"{img_id}\n")
        print(f"\n訓練集 ID 已儲存至: {train_file}")
        
        # 儲存驗證集 ID
        val_file = output_path / 'val_ids.txt'
        with open(val_file, 'w') as f:
            for img_id in val_ids:
                f.write(f"{img_id}\n")
        print(f"驗證集 ID 已儲存至: {val_file}")
    
    def load_test_images(self):
        """
        載入測試集影像列表
        
        Returns:
            list: 測試集影像 ID
        """
        test_images = sorted(self.test_dir.glob('*.png'))
        test_ids = [img.stem for img in test_images]
        
        print(f"\n找到 {len(test_ids)} 張測試影像")
        print(f"測試影像範圍: {test_ids[0]} ~ {test_ids[-1]}")
        
        return test_ids


def main():
    """主程式"""
    
    # 1. 建立載入器
    loader = DroneDatasetLoader(train_dir='train', test_dir='test')
    
    # 2. 載入訓練集標註
    print("正在載入訓練集標註...")
    train_annotations = loader.load_all_train_annotations()
    
    # 3. 顯示統計資訊
    stats = loader.get_dataset_statistics(train_annotations)
    loader.print_statistics(stats)
    
    # 4. 執行 Stratified Split (9:1)
    print("\n執行 Stratified Split (90% train, 10% val)...")
    train_ids, val_ids = loader.stratified_split(
        train_annotations,
        test_size=0.1,
        random_state=42
    )
    
    # 5. 儲存切割結果
    loader.save_split_to_file(train_ids, val_ids, output_dir='splits')
    
    # 6. 載入測試集資訊
    test_ids = loader.load_test_images()
    
    # 7. 示範如何使用切割後的資料
    print("\n" + "="*60)
    print("使用範例")
    print("="*60)
    print(f"訓練時使用: train_ids (共 {len(train_ids)} 張)")
    print(f"驗證時使用: val_ids (共 {len(val_ids)} 張)")
    print(f"測試時使用: test_ids (共 {len(test_ids)} 張)")
    
    return train_ids, val_ids, test_ids, train_annotations


if __name__ == "__main__":
    train_ids, val_ids, test_ids, train_annotations = main()