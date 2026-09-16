import os
import glob
import csv
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import cv2
import random
from collections import defaultdict
import argparse


def Train(dataset_yaml_path='Dataset.yaml'):
    """
    訓練 YOLOv11s 模型
    """
    print("="*60)
    print("開始訓練 YOLOv11s 模型")
    print("="*60)
    
    # =========================================================================
    # 清理 GPU 記憶體
    # =========================================================================
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    print("GPU 記憶體已清理\n")
    
    # =========================================================================
    # 使用 yolo11s
    # =========================================================================
    model = YOLO('yolo11s.yaml')
    print("模型已從 yolo11s.yaml 建立 (不含預訓練權重)")
    
    # =========================================================================
    # 訓練參數
    # =========================================================================
    print("開始訓練...")
    results = model.train(
        data=dataset_yaml_path,
        
        # --- 週期 ---
        epochs=400,
        patience=80,
        
        # --- 批次與解析度---
        batch=4,
        imgsz=1280,
        
        # --- 硬體設定 ---
        pretrained=False,
        workers=4,
        device=0,
        amp=True,
        cache=False,
        
        # --- 學習率---
        lr0=0.005,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        
        # --- 資料增強---
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0001,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.5,
        
        # --- 損失函數 ---
        label_smoothing=0.1,
        box=7.5,
        cls=1.5,
        dfl=1.5,  

        # --- 其他 ---
        close_mosaic=15,
        
        # --- 儲存設定 ---
        project='.',
        name='yolo11s',
        save=True,
        save_period=25,
        
        # --- 驗證設定 ---
        val=True,
        plots=False,
    )
    
    print("\n" + "="*60)
    print("訓練完成!")
    print(f"結果儲存於: {os.path.abspath(results.save_dir)}")
    print("="*60 + "\n")
    
    return results


def get_test_images_path(dataset_yaml_path):
    """從 Dataset.yaml 讀取測試影像路徑"""
    try:
        with open(dataset_yaml_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        if 'test' in dataset_config:
            test_path = dataset_config['test']
            if not os.path.isabs(test_path):
                yaml_dir = os.path.dirname(dataset_yaml_path)
                test_path = os.path.join(yaml_dir, test_path)
            return os.path.normpath(test_path)
        else:
            print(f"錯誤: '{dataset_yaml_path}' 中未找到 'test' 路徑")
            return None
    except Exception as e:
        print(f"讀取 {dataset_yaml_path} 時發生錯誤: {e}")
        return None


def apply_wbf(boxes_list, scores_list, labels_list, iou_thr=0.7, skip_box_thr=0.0001):
    """
    應用 Weighted Boxes Fusion
    
    Args:
        boxes_list: list of arrays, 每個 array 的 shape 是 [N, 4], 格式為 [x1, y1, x2, y2] (normalized)
        scores_list: list of arrays, 每個 array 的 shape 是 [N]
        labels_list: list of arrays, 每個 array 的 shape 是 [N]
        iou_thr: WBF 的 IoU 閾值
        skip_box_thr: 跳過低信心框的閾值
    
    Returns:
        boxes: [M, 4] (normalized)
        scores: [M]
        labels: [M]
    """
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=None,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr
    )
    
    return boxes, scores, labels


def visualize_predictions(image_path, boxes, scores, labels, class_names, output_path):
    """
    將預測框繪製在影像上
    
    Args:
        image_path: 原始影像路徑
        boxes: [N, 4] (x1, y1, x2, y2) denormalized coordinates
        scores: [N] 信心分數
        labels: [N] 類別標籤
        class_names: 類別名稱列表
        output_path: 輸出影像路徑
    """
    # 讀取影像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 定義顏色 (BGR format for cv2)
    colors = {
        0: (255, 0, 0),      # car - 紅色
        1: (0, 255, 0),      # hov - 綠色
        2: (0, 255, 255),    # person - 黃色
        3: (255, 0, 255)     # motorcycle - 紫色
    }
    
    # 繪製每個預測框
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        label_int = int(label)
        color = colors.get(label_int, (255, 255, 255))
        
        # 繪製矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 準備文字標籤
        class_name = class_names[label_int] if label_int < len(class_names) else str(label_int)
        text = f"{class_name} {score:.2f}"
        
        # 計算文字背景大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 繪製文字背景
        cv2.rectangle(img, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        
        # 繪製文字 (黑色粗體)
        cv2.putText(img, text, (x1, y1 - 5), font, font_scale, (0, 0, 0), 2)
    
    # 儲存影像
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img_bgr)
    
    return img


def analyze_and_save_cases(predictions_data, test_images_dir, output_dir='visualization_results'):
    """
    分析預測結果並保存成功與失敗案例
    """
    class_names = ['car', 'hov', 'person', 'motorcycle']
    
    if not predictions_data:
        print("警告: 沒有預測數據可供可視化")
        return
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    success_dir = os.path.join(output_dir, 'success_cases')
    failure_dir = os.path.join(output_dir, 'failure_cases')
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failure_dir, exist_ok=True)
    
    # 計算每張影像的檢測統計
    detection_stats = []
    for data in predictions_data:
        scores = data['scores']
        
        # 分三個信心度區間統計
        very_high_conf_mask = scores > 0.5  # 非常高信心度
        high_conf_mask = (scores > 0.3) & (scores <= 0.5)  # 高信心度
        low_conf_mask = scores <= 0.3  # 低信心度
        
        num_very_high = np.sum(very_high_conf_mask)
        num_high = np.sum(high_conf_mask)
        num_low = np.sum(low_conf_mask)
        num_total = len(scores)
        
        # 計算平均信心度（只計算 > 0.3 的）
        high_conf_scores = scores[scores > 0.3]
        avg_high_conf = np.mean(high_conf_scores) if len(high_conf_scores) > 0 else 0
        
        # 計算類別分佈（只計算高信心度）
        class_counts = defaultdict(int)
        for label, score in zip(data['labels'], scores):
            if score > 0.3:
                class_counts[int(label)] += 1
        
        # 計算「質量分數」= 高信心度比例 * 高信心度平均值
        quality_score = (num_very_high + num_high) / max(num_total, 1) * avg_high_conf if num_total > 0 else 0
        
        # 計算「誤檢嚴重度」= 低信心度數量 / 高信心度數量
        false_positive_severity = num_low / max(num_very_high + num_high, 1)
        
        detection_stats.append({
            'data': data,
            'num_total': num_total,
            'num_very_high': num_very_high,
            'num_high': num_high,
            'num_low': num_low,
            'avg_high_conf': avg_high_conf,
            'class_counts': class_counts,
            'quality_score': quality_score,
            'false_positive_severity': false_positive_severity
        })
    
    # 成功案例：質量分數最高的
    sorted_by_success = sorted(
        detection_stats, 
        key=lambda x: (
            x['quality_score'],      # 質量分數（結合高信心度比例和平均值）
            x['num_very_high'],      # 非常高信心度數量
            -x['false_positive_severity']  # 誤檢越少越好
        ), 
        reverse=True
    )
    
    # 失敗案例：質量分數最低或誤檢最嚴重的
    sorted_by_failure = sorted(
        detection_stats,
        key=lambda x: (
            x['quality_score'],      # 質量分數低
            -x['false_positive_severity']  # 誤檢嚴重
        )
    )
    
    print("\n" + "="*60)
    print("保存可視化結果")
    print("="*60)
    
    # 保存前3個成功案例
    print("\n成功案例 (高質量檢測):")
    for i, stat in enumerate(sorted_by_success[:3]):
        data = stat['data']
        output_path = os.path.join(success_dir, f'success_{i+1}.jpg')
        
        visualize_predictions(
            data['image_path'],
            data['boxes'],
            data['scores'],
            data['labels'],
            class_names,
            output_path
        )
        
        print(f"  {i+1}. {os.path.basename(data['image_path'])}")
        print(f"     總檢測: {stat['num_total']}, 非常高信心(>0.5): {stat['num_very_high']}, "
              f"高信心(0.3-0.5): {stat['num_high']}, 低信心(≤0.3): {stat['num_low']}")
        print(f"     高信心平均: {stat['avg_high_conf']:.3f}, 質量分數: {stat['quality_score']:.3f}")
        print(f"     類別分佈: {dict(stat['class_counts'])}")
    
    # 保存前3個失敗案例
    print("\n失敗案例 (低質量或嚴重誤檢):")
    for i, stat in enumerate(sorted_by_failure[:3]):
        data = stat['data']
        output_path = os.path.join(failure_dir, f'failure_{i+1}.jpg')
        
        visualize_predictions(
            data['image_path'],
            data['boxes'],
            data['scores'],
            data['labels'],
            class_names,
            output_path
        )
        
        print(f"  {i+1}. {os.path.basename(data['image_path'])}")
        print(f"     總檢測: {stat['num_total']}, 非常高信心(>0.5): {stat['num_very_high']}, "
              f"高信心(0.3-0.5): {stat['num_high']}, 低信心(≤0.3): {stat['num_low']}")
        print(f"     高信心平均: {stat['avg_high_conf']:.3f}, 質量分數: {stat['quality_score']:.3f}")
        print(f"     誤檢嚴重度: {stat['false_positive_severity']:.2f}")
        print(f"     類別分佈: {dict(stat['class_counts'])}")
    
    print(f"\n可視化結果已保存至: {output_dir}")
    print("="*60 + "\n")


def generate_submission_with_wbf(trained_model_path, test_images_dir, output_csv):
    """
    使用 WBF + TTA 生成預測檔案
    
    Args:
        trained_model_path: 訓練好的模型路徑
        test_images_dir: 測試影像資料夾
        output_csv: 輸出 CSV 檔案名稱
    """
    # 清理記憶體
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    if not Path(trained_model_path).exists():
        print(f"錯誤: 找不到模型 {trained_model_path}")
        return
    
    print("="*60)
    print("開始生成預測檔案")
    print("="*60)
    print(f"載入模型: {trained_model_path}")
    model = YOLO(trained_model_path)
    
    if not Path(test_images_dir).exists():
        print(f"錯誤: 找不到測試影像資料夾 {test_images_dir}")
        return
    
    print(f"尋找測試影像: {test_images_dir}")
    image_files = sorted(
        glob.glob(os.path.join(test_images_dir, '*.png')) +
        glob.glob(os.path.join(test_images_dir, '*.jpg')) +
        glob.glob(os.path.join(test_images_dir, '*.jpeg'))
    )
    
    if not image_files:
        print(f"錯誤: 未找到任何影像")
        return
    
    print(f"找到 {len(image_files)} 張測試影像")
    print(f"配置:")
    print(f"  - 解析度: 1280")
    print(f"  - TTA: True")
    print(f"  - WBF: True")
    print(f"  - Confidence: 0.001")
    print(f"  - NMS IoU: 0.6")
    print(f"  - WBF IoU: 0.7\n")
    
    header = ['Image_ID', 'PredictionString']
    rows = []
    predictions_data = []  # 用於收集可視化數據

    print("開始預測...")
    
    for idx, img_path in enumerate(image_files):
        if (idx + 1) % 50 == 0:
            print(f"處理進度: {idx + 1}/{len(image_files)}")
        
        # 獲取影像尺寸
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # =========================================================
        # 單一尺度 + TTA + WBF
        # =========================================================
        all_boxes = []
        all_scores = []
        all_labels = []
        
        # TTA: 進行兩次預測並融合
        for _ in range(2):
            results = model.predict(
                img_path,
                imgsz=1280,
                conf=0.001,
                iou=0.6,
                agnostic_nms=False,
                max_det=800,
                augment=True,
                verbose=False,
                device=0
            )[0]
            
            boxes = results.boxes
            if boxes is not None and len(boxes) > 0:
                # 獲取座標 (denormalized)
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                
                # 轉換為 normalized 座標
                normalized_boxes = xyxy.copy()
                normalized_boxes[:, [0, 2]] /= img_width
                normalized_boxes[:, [1, 3]] /= img_height
                
                # 確保座標在 [0, 1] 範圍內
                normalized_boxes = np.clip(normalized_boxes, 0, 1)
                
                all_boxes.append(normalized_boxes)
                all_scores.append(confs)
                all_labels.append(classes)
        
        # 應用 WBF
        if len(all_boxes) > 0:
            fused_boxes, fused_scores, fused_labels = apply_wbf(
                all_boxes,
                all_scores,
                all_labels,
                iou_thr=0.7,
                skip_box_thr=0.0001
            )
            
            # 轉換回 denormalized 座標
            fused_boxes[:, [0, 2]] *= img_width
            fused_boxes[:, [1, 3]] *= img_height
            
            # 格式化輸出
            prediction_strings = []
            for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
                x1, y1, x2, y2 = box
                bb_left = x1
                bb_top = y1
                bb_width = x2 - x1
                bb_height = y2 - y1
                
                pred_str = f"{score:.6f} {bb_left:.2f} {bb_top:.2f} {bb_width:.2f} {bb_height:.2f} {int(label)}"
                prediction_strings.append(pred_str)
            
            final_prediction_string = " ".join(prediction_strings)
        else:
            final_prediction_string = ""
        
        image_id = idx + 1
        rows.append([image_id, final_prediction_string])

        # 收集預測數據用於可視化
        if len(all_boxes) > 0:
            predictions_data.append({
                'image_path': img_path,
                'boxes': fused_boxes,
                'scores': fused_scores,
                'labels': fused_labels
            })
        else:
            # 也記錄沒有檢測結果的影像（這些可能是失敗案例）
            predictions_data.append({
                'image_path': img_path,
                'boxes': np.array([]),
                'scores': np.array([]),
                'labels': np.array([])
            })

    print(f"\n寫入預測檔案: {output_csv}")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    # 生成可視化結果
    print("\n生成可視化結果...")
    analyze_and_save_cases(predictions_data, test_images_dir)

    print("="*60)
    print(f"預測檔案 {output_csv} 已成功產生!")
    print("="*60 + "\n")


def main():
    """主程式"""
    dataset_yaml_file = 'Dataset.yaml'
    
    # =========================================================================
    # 訓練模型
    # =========================================================================
    print("\n" + "="*60)
    print("訓練模型")
    print("="*60 + "\n")
    
    try:
        results = Train(dataset_yaml_file)
        
        # =========================================================================
        # 生成預測檔案
        # =========================================================================
        print("\n" + "="*60)
        print("生成預測檔案")
        print("="*60 + "\n")
        
        trained_model_path = os.path.join(results.save_dir, 'weights/best.pt')
        test_images_dir = get_test_images_path(dataset_yaml_file)
        
        if test_images_dir:
            output_csv = 'submission.csv'
            
            # 使用 WBF + TTA 生成預測檔案
            generate_submission_with_wbf(
                trained_model_path,
                test_images_dir,
                output_csv
            )
            
            print("\n" + "="*60)
            print("全部完成!")
            print("="*60)
            print(f"訓練結果: {results.save_dir}")
            print(f"預測檔案: {output_csv}")
            print("="*60 + "\n")
        else:
            print("無法產生預測檔案")
    
    except Exception as e:
        print(f"執行過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()


# =========================================================================
# 只重新生成預測檔案 (不重新訓練)
# =========================================================================
def Inference():
    """只生成預測檔案,不訓練"""
    dataset_yaml_file = 'Dataset.yaml'
    
    # 指定訓練好的模型路徑
    trained_model_path = 'yolo11s/weights/best.pt'

    if not Path(trained_model_path).exists():
        print(f"錯誤: 找不到模型 {trained_model_path}")
        print("請檢查路徑是否正確")
        return
    
    test_images_dir = get_test_images_path(dataset_yaml_file)
    
    if test_images_dir:
        generate_submission_with_wbf(
            trained_model_path,
            test_images_dir,
            'submission.csv'
        )


if __name__ == "__main__":
    # 設定 argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        required=True,  # 使用者必須提供 --mode 參數
        choices=['train', 'inference'],
        help="選擇執行模式: 'train' (訓練並推論) 或 'inference' (僅推論)"
    )
    args = parser.parse_args()

    # 根據模式執行
    if args.mode == "train":
        print("執行模式: train")
        main()
    elif args.mode == "inference":
        print("執行模式: inference")
        Inference()