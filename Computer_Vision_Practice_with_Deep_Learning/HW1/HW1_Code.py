import os
import csv
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import resnet50
from torchvision.ops import box_iou
import numpy as np
import random
import math
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from ensemble_boxes import weighted_boxes_fusion
import argparse
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ========= Dataset Split Function =========
def split_dataset(img_dir, gt_file, out_dir, val_ratio=0.2, seed=42):
    """Split dataset into train and validation sets"""
    random.seed(seed)
    
    # Read all image files
    all_imgs = sorted(os.listdir(img_dir))
    total = len(all_imgs)
    val_size = int(total * val_ratio)
    
    val_imgs = set(random.sample(all_imgs, val_size))
    train_imgs = set(all_imgs) - val_imgs
    
    print(f"Total {total} images: train={len(train_imgs)}, val={len(val_imgs)}")
    
    # Create output directories
    train_img_dir = os.path.join(out_dir, "train/img")
    val_img_dir = os.path.join(out_dir, "val/img")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    
    # Read annotations
    with open(gt_file, "r", encoding='utf-8') as f:
        lines = f.readlines()
    
    train_gt = []
    val_gt = []
    
    for line in lines:
        frame = int(line.strip().split(",")[0])
        fname = f"{frame:08d}.jpg"
        if fname in val_imgs:
            val_gt.append(line)
        else:
            train_gt.append(line)
    
    # Write new gt.txt files
    with open(os.path.join(out_dir, "train/gt.txt"), "w", encoding='utf-8') as f:
        f.writelines(train_gt)
    
    with open(os.path.join(out_dir, "val/gt.txt"), "w", encoding='utf-8') as f:
        f.writelines(val_gt)
    
    # Copy images
    import shutil
    for fname in train_imgs:
        shutil.copy(os.path.join(img_dir, fname), os.path.join(train_img_dir, fname))
    
    for fname in val_imgs:
        shutil.copy(os.path.join(img_dir, fname), os.path.join(val_img_dir, fname))
    
    print("Dataset split complete!\n")


# ========= Custom Data Augmentation =========
class Compose:
    """Compose multiple transforms"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert to Tensor"""
    def __call__(self, image, target):
        image = T.functional.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    """Random horizontal flip (supports PIL and Tensor)"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = T.functional.hflip(image)
            if target is not None:
                if isinstance(image, Image.Image):
                    width, height = image.size
                else:
                    _, height, width = image.shape
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


class RandomVerticalFlip:
    """Random vertical flip (supports PIL and Tensor)"""
    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = T.functional.vflip(image)
            if target is not None:
                if isinstance(image, Image.Image):
                    width, height = image.size
                else:
                    _, height, width = image.shape
                boxes = target["boxes"].clone()
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
                target["boxes"] = boxes
        return image, target


class ColorJitter:
    """Color jitter"""
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1):
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class GaussianBlur:
    def __init__(self, kernel_size=5, prob=0.2):
        self.kernel_size = kernel_size
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = T.functional.gaussian_blur(image, kernel_size=self.kernel_size)
        return image, target


class RandomGrayscale:
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = T.functional.rgb_to_grayscale(image, num_output_channels=3)
        return image, target


# ========= Dataset Loading =========
class PigDataset(Dataset):
    def __init__(self, img_dir, gt_file=None, transform=None, min_size=1.0):
        self.img_dir = img_dir
        self.transform = transform
        self.has_label = gt_file is not None
        self.min_size = min_size

        self.img_files = sorted(os.listdir(img_dir))
        self.labels = {}

        if self.has_label:
            try:
                with open(gt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        frame, x, y, w, h = map(float, line.strip().split(','))
                        frame = int(frame)
                        if frame not in self.labels:
                            self.labels[frame] = []
                        self.labels[frame].append([x, y, w, h])
            except UnicodeDecodeError:
                with open(gt_file, 'r', encoding='latin-1') as f:
                    for line in f:
                        frame, x, y, w, h = map(float, line.strip().split(','))
                        frame = int(frame)
                        if frame not in self.labels:
                            self.labels[frame] = []
                        self.labels[frame].append([x, y, w, h])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = Image.open(img_path).convert("RGB")
        img_id = int(os.path.splitext(self.img_files[idx])[0])

        if not self.has_label:
            if self.transform:
                img, _ = self.transform(img, None)
            else:
                img = T.functional.to_tensor(img)
            return img, img_id

        raw_boxes = torch.as_tensor(self.labels[img_id], dtype=torch.float32)
        boxes = raw_boxes.clone()
        boxes[:, 2:] += boxes[:, :2]

        keep = (boxes[:, 2] - boxes[:, 0] > self.min_size) & \
               (boxes[:, 3] - boxes[:, 1] > self.min_size)
        boxes = boxes[keep]

        if boxes.numel() == 0:
            labels = torch.zeros((0,), dtype=torch.int64)
            target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}
            if self.transform:
                img, target = self.transform(img, target)
            return img, target

        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}

        if self.transform:
            img, target = self.transform(img, target)

        return img, target


# ========= Get Transform Functions =========
def get_train_transform(augment=True):
    if augment:
        return Compose([
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            RandomHorizontalFlip(prob=0.5),
            RandomVerticalFlip(prob=0.3),
            GaussianBlur(kernel_size=5, prob=0.2),
            RandomGrayscale(prob=0.1),
            ToTensor(),
        ])
    else:
        return Compose([ToTensor()])


def get_test_transform():
    return Compose([ToTensor()])


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        # Classification Header
        self.cls_score = nn.Linear(in_channels, num_classes)
        # Bounding Box Regression Head
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas
    

# ========= Build Model =========
def get_model(num_classes=2):
    """Build Faster R-CNN using ResNet50_v2 backbone pretrained on ImageNet"""

    # Establish Faster R-CNN (without loading detection pre-trained weights)
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    # Load ResNet50 v2 (ImageNet1K_V2) classification pre-trained weights
    resnet_pretrained = resnet50(weights="IMAGENET1K_V2")
    backbone_state_dict = resnet_pretrained.state_dict()

    # Map the backbone weights to the backbone of the detection model
    model_state_dict = model.state_dict()
    pretrained_dict = {}
    for k, v in backbone_state_dict.items():
        new_key = f"backbone.body.{k}"
        if new_key in model_state_dict:
            pretrained_dict[new_key] = v

    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)

    # Modify the output layer to ensure the number of classes aligns with your task.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ========= Calculate mAP =========
def calculate_map(predictions, ground_truths, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    for iou in iou_thresholds:
        aps.append(calculate_ap_at_iou(predictions, ground_truths, iou))
    return np.mean(aps)


def calculate_ap_at_iou(predictions, ground_truths, iou_threshold):
    all_scores = []
    all_tp = []
    all_fp = []
    num_gt = 0

    for img_id in ground_truths.keys():
        gt_boxes = ground_truths[img_id]['boxes']
        num_gt += len(gt_boxes)

        if img_id not in predictions:
            continue

        pred_boxes = predictions[img_id]['boxes']
        pred_scores = predictions[img_id]['scores']

        if len(pred_boxes) == 0:
            continue

        if len(gt_boxes) > 0:
            ious = box_iou(pred_boxes, gt_boxes)
            gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

            for i in range(len(pred_boxes)):
                max_iou, max_idx = ious[i].max(dim=0)
                if max_iou >= iou_threshold and not gt_matched[max_idx]:
                    all_tp.append(1)
                    all_fp.append(0)
                    gt_matched[max_idx] = True
                else:
                    all_tp.append(0)
                    all_fp.append(1)
                all_scores.append(pred_scores[i].item())
        else:
            all_tp.extend([0] * len(pred_boxes))
            all_fp.extend([1] * len(pred_boxes))
            all_scores.extend(pred_scores.tolist())

    if len(all_scores) == 0 or num_gt == 0:
        return 0.0

    indices = np.argsort(all_scores)[::-1]
    tp = np.array(all_tp)[indices]
    fp = np.array(all_fp)[indices]

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])

    return ap


# ========= Validation Function =========
def validate(model, dataloader, device):
    model.eval()
    predictions = {}
    ground_truths = {}

    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)

            for target, output in zip(targets, outputs):
                img_id = int(target['image_id'].item())
                ground_truths[img_id] = {'boxes': target['boxes'].cpu()}
                predictions[img_id] = {'boxes': output['boxes'].cpu(),
                                       'scores': output['scores'].cpu()}

    map_score = calculate_map(predictions, ground_truths)
    return map_score


# ========= Train One Epoch =========
def train_one_epoch(model, optimizer, dataloader, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for imgs, targets in dataloader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / (num_batches if num_batches > 0 else 1)
    print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    return avg_loss


# ========= WBF Function =========

def apply_wbf_single_image(boxes_list, scores_list, labels_list, image_size, iou_thr=0.55, skip_box_thr=0.0001):
    """
    Apply Weighted Boxes Fusion for a single image
    
    Args:
        boxes_list: list of torch tensors, each [N, 4] in pixel coordinates
        scores_list: list of torch tensors, each [N]
        labels_list: list of torch tensors, each [N]
        image_size: tuple (height, width)
        iou_thr: IoU threshold for WBF
        skip_box_thr: minimum score threshold
    
    Returns:
        boxes: torch tensor [M, 4] in pixel coordinates
        scores: torch tensor [M]
        labels: torch tensor [M]
    """
    H, W = image_size
    
    # Convert to normalized coordinates (0-1)
    boxes_list_norm = []
    scores_list_norm = []
    labels_list_norm = []
    
    for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
        if len(boxes) == 0:
            continue
            
        boxes_norm = boxes.clone()
        boxes_norm[:, [0, 2]] /= W
        boxes_norm[:, [1, 3]] /= H
        
        # Clamp to [0, 1]
        boxes_norm = torch.clamp(boxes_norm, 0, 1)
        
        boxes_list_norm.append(boxes_norm.cpu().numpy().tolist())
        scores_list_norm.append(scores.cpu().numpy().tolist())
        labels_list_norm.append(labels.cpu().numpy().tolist())
    
    if len(boxes_list_norm) == 0:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.int64)
    
    # Apply WBF
    boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(
        boxes_list_norm,
        scores_list_norm,
        labels_list_norm,
        weights=None,  # Equal weights
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr
    )
    
    # Convert back to pixel coordinates
    boxes_wbf = torch.tensor(boxes_wbf, dtype=torch.float32)
    boxes_wbf[:, [0, 2]] *= W
    boxes_wbf[:, [1, 3]] *= H
    
    scores_wbf = torch.tensor(scores_wbf, dtype=torch.float32)
    labels_wbf = torch.tensor(labels_wbf, dtype=torch.int64)
    
    return boxes_wbf, scores_wbf, labels_wbf


# ========= Predict and Export =========
def predict_and_export(model, dataloader, device, out_csv="submission.csv", score_thresh=0.3, use_tta=True, wbf_iou=0.55):
    model.eval()
    results = {}
    print("\nStarting inference...")
    print(f"TTA enabled: {use_tta}, WBF IoU threshold: {wbf_iou}, Score threshold: {score_thresh}")
    
    with torch.no_grad():
        for batch_idx, (imgs, img_ids) in enumerate(dataloader):
            imgs = [img.to(device) for img in imgs]
            
            for img_tensor, img_id in zip(imgs, img_ids):
                img_id = int(img_id)
                _, H, W = img_tensor.shape
                
                all_boxes = []
                all_scores = []
                all_labels = []
                
                # Original image prediction
                output = model([img_tensor])[0]
                all_boxes.append(output['boxes'].cpu())
                all_scores.append(output['scores'].cpu())
                all_labels.append(output.get('labels', torch.ones(len(output['boxes']), dtype=torch.int64)).cpu())
                
                if use_tta:
                    # Horizontal flip TTA
                    img_hflip = T.functional.hflip(img_tensor)
                    output_hflip = model([img_hflip])[0]
                    boxes_hflip = output_hflip['boxes'].cpu()
                    # Flip boxes back
                    boxes_hflip[:, [0, 2]] = W - boxes_hflip[:, [2, 0]]
                    all_boxes.append(boxes_hflip)
                    all_scores.append(output_hflip['scores'].cpu())
                    all_labels.append(output_hflip.get('labels', torch.ones(len(boxes_hflip), dtype=torch.int64)).cpu())
                    
                    # Vertical flip TTA
                    img_vflip = T.functional.vflip(img_tensor)
                    output_vflip = model([img_vflip])[0]
                    boxes_vflip = output_vflip['boxes'].cpu()
                    # Flip boxes back
                    boxes_vflip[:, [1, 3]] = H - boxes_vflip[:, [3, 1]]
                    all_boxes.append(boxes_vflip)
                    all_scores.append(output_vflip['scores'].cpu())
                    all_labels.append(output_vflip.get('labels', torch.ones(len(boxes_vflip), dtype=torch.int64)).cpu())
                
                # Apply WBF
                boxes_fused, scores_fused, labels_fused = apply_wbf_single_image(
                    all_boxes, all_scores, all_labels,
                    image_size=(H, W),
                    iou_thr=wbf_iou,
                    skip_box_thr=0.0001
                )
                
                # Format predictions
                preds = []
                for box, score in zip(boxes_fused, scores_fused):
                    if score < score_thresh:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    x1 = max(0, min(x1, W - 1))
                    y1 = max(0, min(y1, H - 1))
                    x2 = max(0, min(x2, W - 1))
                    y2 = max(0, min(y2, H - 1))
                    w, h = x2 - x1, y2 - y1
                    if w <= 1 or h <= 1:
                        continue
                    preds.append(f"{score:.4f} {x1:.1f} {y1:.1f} {w:.1f} {h:.1f} 0")
                
                results[img_id] = " ".join(preds)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Progress: {batch_idx + 1}/{len(dataloader)} batches")
    
    print(f"\nSaving results to {out_csv}...")
    with open(out_csv, "w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Image_ID", "PredictionString"])
        for img_id in sorted(results.keys()):
            writer.writerow([img_id, results[img_id]])
    print("Inference complete! Results saved.")


# ========= Final Mode Execution =========
def run_final_mode(device, args):
    print("[FINAL MODE] Using all training data (generating submission)\n")

    train_img_dir = "./train/img"
    train_gt = "./train/gt.txt"
    test_img_dir = "./test/img"

    train_transform = get_train_transform(augment=True)
    test_transform = get_test_transform()

    print("Loading datasets...")
    train_dataset = PigDataset(train_img_dir, train_gt, train_transform)
    test_dataset = PigDataset(test_img_dir, None, test_transform)
    print(f"Training set: {len(train_dataset)} images")
    print(f"Test set: {len(test_dataset)} images\n")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(num_classes=2).to(device)

    if not os.path.exists('best_model.pth'):
        raise FileNotFoundError("best_model.pth not found")

    print("Loading best model from dev phase: best_model.pth...")

    # Load 'best_model.pth'
    if args.mode == 'dev': # dev mode
        checkpoint = torch.load('best_model.pth', map_location=device)
    else:  # final mode
        checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best dev model (Epoch {checkpoint.get('epoch', '?')}, mAP={checkpoint.get('map', -1):.4f})")

    params = [p for p in model.parameters() if p.requires_grad]
    

    # Final mode
    optimizer = torch.optim.AdamW(params, lr=0.0005, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


    print("Starting final training...\n")
    for epoch in range(args.final_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()

    torch.save(model.state_dict(), "final_model.pth")
    print("Final model saved to final_model.pth")

    predict_and_export(
        model, 
        test_loader, 
        device, 
        "submission.csv",
        score_thresh=0.3, 
        use_tta=True, 
        wbf_iou=0.55 
    )


# Use color (green/red) to distinguish between high and low confidence scores
def visualize_test_predictions(model, dataloader, device, num_images=6, score_thresh=0.3, high_conf_thresh=0.7):
    
    model.eval()
    cases = []
    
    with torch.no_grad():
        for imgs, img_ids in dataloader:
            if len(cases) >= num_images * 2:
                break
                
            imgs_tensor = [img.to(device) for img in imgs]
            outputs = model(imgs_tensor)
            
            for img, img_id, output in zip(imgs, img_ids, outputs):
                if len(cases) >= num_images * 2:
                    break
                    
                img_id = int(img_id)
                pred_boxes = output['boxes'].cpu()
                pred_scores = output['scores'].cpu()
                
                # Filter low-scoring predictions
                keep = pred_scores > score_thresh
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                
                if len(pred_scores) == 0:
                    continue
                
                # Separate High and Low Confidence Predictions
                high_conf_mask = pred_scores >= high_conf_thresh
                low_conf_mask = pred_scores < high_conf_thresh
                
                high_conf_boxes = pred_boxes[high_conf_mask]
                high_conf_scores = pred_scores[high_conf_mask]
                low_conf_boxes = pred_boxes[low_conf_mask]
                low_conf_scores = pred_scores[low_conf_mask]
                
                # Calculate Quality Metrics
                avg_score = pred_scores.mean().item()
                high_conf_ratio = len(high_conf_boxes) / len(pred_boxes)
                
                cases.append({
                    'img': img.cpu(),
                    'high_conf_boxes': high_conf_boxes,
                    'high_conf_scores': high_conf_scores,
                    'low_conf_boxes': low_conf_boxes,
                    'low_conf_scores': low_conf_scores,
                    'img_id': img_id,
                    'num_preds': len(pred_boxes),
                    'num_high': len(high_conf_boxes),
                    'num_low': len(low_conf_boxes),
                    'avg_score': avg_score,
                    'high_conf_ratio': high_conf_ratio
                })
    
    if not cases:
        print("No predictions to visualize")
        return
    
    # Sort by average score and high confidence ratio
    cases.sort(key=lambda x: (x['avg_score'], x['high_conf_ratio']), reverse=True)
    
    success_cases = cases[:num_images//2]
    failure_cases = cases[-(num_images//2):]
    
    # Plot Success Cases
    fig, axes = plt.subplots(1, len(success_cases), figsize=(5*len(success_cases), 5))
    if len(success_cases) == 1:
        axes = [axes]
    fig.suptitle('Success Cases (Better Predictions)', fontsize=16, fontweight='bold')
    
    for idx, case in enumerate(success_cases):
        ax = axes[idx]
        img_np = case['img'].permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        
        # Plot the high confidence box (green)
        for box, score in zip(case['high_conf_boxes'], case['high_conf_scores']):
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{score:.2f}', color='green', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot the low confidence box (red)
        for box, score in zip(case['low_conf_boxes'], case['low_conf_scores']):
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{score:.2f}', color='red', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'ID: {case["img_id"]}\nHigh conf (green): {case["num_high"]}, Low conf (red): {case["num_low"]}\nAvg: {case["avg_score"]:.3f}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('success_cases.png', dpi=150, bbox_inches='tight')
    print(f"Saved success cases to success_cases.png")
    plt.close()
    
    # Plot Failure Cases
    fig, axes = plt.subplots(1, len(failure_cases), figsize=(5*len(failure_cases), 5))
    if len(failure_cases) == 1:
        axes = [axes]
    fig.suptitle('Failure Cases (Worse Predictions)', fontsize=16, fontweight='bold')
    
    for idx, case in enumerate(failure_cases):
        ax = axes[idx]
        img_np = case['img'].permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        
        # Plot the high confidence box (green)
        for box, score in zip(case['high_conf_boxes'], case['high_conf_scores']):
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{score:.2f}', color='green', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot the low confidence box (red)
        for box, score in zip(case['low_conf_boxes'], case['low_conf_scores']):
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{score:.2f}', color='red', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'ID: {case["img_id"]}\nHigh conf (green): {case["num_high"]}, Low conf (red): {case["num_low"]}\nAvg: {case["avg_score"]:.3f}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('failure_cases.png', dpi=150, bbox_inches='tight')
    print(f"Saved failure cases to failure_cases.png")
    plt.close()


# ========= Main Program =========
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='dev', choices=['dev', 'final', 'inference'],
                        help='dev: use split dataset for development; final: use all data and generate submission; inference: load final_model.pth and generate submission')
    parser.add_argument('--epochs', type=int, default=70, help='number of training epochs')
    parser.add_argument('--final-epochs', type=int, default=70, help='number of training epochs for final mode')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("HW1 - Object Detection (Pig Detection)")
    print("Only loading backbone pretrained weights")
    print(f"Mode: {args.mode.upper()}")
    print("=" * 60 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    if args.mode == 'dev':
        print("[DEV MODE] Using split training and validation sets\n")

        # Check if split data exists, if not, create it
        if not os.path.exists("./split_data/train/gt.txt") or \
           not os.path.exists("./split_data/val/gt.txt"):
            print("Split data not found. Creating train/val split...")
            split_dataset(
                img_dir="./train/img",
                gt_file="./train/gt.txt",
                out_dir="./split_data",
                val_ratio=0.2,
                seed=42
            )
        else:
            print("Using existing split data in ./split_data/\n")

        train_img_dir = "./split_data/train/img"
        train_gt = "./split_data/train/gt.txt"
        val_img_dir = "./split_data/val/img"
        val_gt = "./split_data/val/gt.txt"

        train_transform = get_train_transform(augment=True)
        val_transform = get_test_transform()

        print("Loading datasets...")
        train_dataset = PigDataset(train_img_dir, train_gt, train_transform)
        val_dataset = PigDataset(val_img_dir, val_gt, val_transform)
        print(f"Training set: {len(train_dataset)} images")
        print(f"Validation set: {len(val_dataset)} images\n")

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

        model = get_model(num_classes=2).to(device)

        params = [p for p in model.parameters() if p.requires_grad]


        # Dev mode
        optimizer = torch.optim.AdamW(params, lr=0.0005, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        best_map = 0.0
        best_epoch = 0
        patience = 10
        no_improve_count = 0

        print(f"{'Epoch':<8} {'Train Loss':<12} {'Val mAP50:95':<15} {'Best mAP':<12} {'Status'}")
        print("-" * 70)

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
            lr_scheduler.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print("  -> Calculating mAP on validation set...")
                val_map = validate(model, val_loader, device)

                status = ""
                if val_map > best_map:
                    best_map = val_map
                    best_epoch = epoch + 1
                    no_improve_count = 0

                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'map': val_map,
                        'loss': train_loss
                    }, 'best_model.pth')
                    status = "New best model! Saved"
                else:
                    no_improve_count += 1
                    status = f"({no_improve_count}/{patience} no improvement)"

                print(f"Epoch {epoch+1:<3} Loss: {train_loss:.4f}   mAP: {val_map:.4f}   Best: {best_map:.4f}   {status}")

                if no_improve_count >= patience:
                    print(f"\nEarly stopping: {patience} epochs without improvement")
                    break
            else:
                print(f"Epoch {epoch+1:<3} Loss: {train_loss:.4f}   (skipping validation)")

        print("\n" + "=" * 60)
        print(f"Dev mode training complete!")
        print(f"Best model: Epoch {best_epoch}, mAP50:95 = {best_map:.4f}")
        print(f"Model saved to best_model.pth")
        print("=" * 60)
        print("\nTo run final training, use:")
        print("  python3 HW1.py --mode final")

    elif args.mode == 'inference':
        print("[INFERENCE MODE] Loading final_model.pth and generating submission\n")
        
        test_img_dir = "./test/img"
        test_transform = get_test_transform()
        
        print("Loading test dataset...")
        test_dataset = PigDataset(test_img_dir, None, test_transform)
        print(f"Test set: {len(test_dataset)} images\n")
        
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, 
                                collate_fn=lambda x: tuple(zip(*x)))
        
        model = get_model(num_classes=2).to(device)
        
        if not os.path.exists('final_model.pth'):
            raise FileNotFoundError("final_model.pth not found")
        
        print("Loading final_model.pth...")
        model.load_state_dict(torch.load('final_model.pth', map_location=device, weights_only=False))
        print("Model loaded successfully\n")
        
        predict_and_export(
            model, 
            test_loader, 
            device, 
            "submission.csv",
            score_thresh=0.3, 
            use_tta=True, 
            wbf_iou=0.55
        )

        # ===== Visualize Predictions =====
        print("\nGenerating visualization of test predictions...")
        visualize_test_predictions(model, test_loader, device, num_images=6, score_thresh=0.3, high_conf_thresh=0.7)
        # ===================
        
        print("\nSubmission file saved to: submission.csv")

    else:
        run_final_mode(device, args)


if __name__ == "__main__":
    main()