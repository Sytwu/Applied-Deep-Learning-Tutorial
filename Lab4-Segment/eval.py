"""
評估模組 - 語義分割評估指標
包含 Dice Score, mIoU 等分割任務常用指標
"""

import torch
import numpy as np
from typing import Tuple


def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    計算 Dice Score (F1 Score)
    
    Args:
        pred: 預測的分割 mask (H, W) 或 (C, H, W)
        target: 真實的分割 mask (H, W)
        smooth: 平滑項，防止除以零
    
    Returns:
        Dice Score, 範圍 [0, 1]
    """
    # TODO: Student Implementation
    # Dice Score = 2 * |預測 ∩ 真實| / (|預測| + |真實|)
    
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice.item()


def calculate_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    計算像素準確率
    
    Args:
        pred: 預測的分割 mask  (H, W)
        target: 真實的分割 mask (H, W)
    
    Returns:
        像素準確率, 範圍 [0, 1]
    """
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total


def calculate_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """
    計算 mIoU (mean Intersection over Union)
    
    Args:
        pred: 預測的分割 mask (H, W)
        target: 真實的分割 mask (H, W)
        num_classes: 類別數量
    
    Returns:
        mIoU, 範圍 [0, 1]
    """
    # TODO: Student Implementation
    # 對每個類別計算 IoU，然後取平均
    
    ious = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        
        if union == 0:
            # 該類別不存在
            continue
        
        iou = intersection / union
        ious.append(iou)
    
    if len(ious) == 0:
        return 0.0
    
    return np.mean(ious)


def visualize_segmentation(
    image: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor = None,
    num_classes: int = 3
):
    """
    視覺化分割結果
    
    Args:
        image: 原始影像 (3, H, W)
        pred_mask: 預測的分割 mask (H, W)
        gt_mask: 真實的分割 mask (H, W) (可選)
        num_classes: 類別數量
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # 定義顏色映射
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
    cmap = mcolors.ListedColormap(colors[:num_classes])
    
    if gt_mask is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始影像
        axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title('原始影像')
        axes[0].axis('off')
        
        # 真實 mask
        axes[1].imshow(gt_mask.cpu().numpy(), cmap=cmap)
        axes[1].set_title('真實分割')
        axes[1].axis('off')
        
        # 預測 mask
        axes[2].imshow(pred_mask.cpu().numpy(), cmap=cmap)
        axes[2].set_title('預測分割')
        axes[2].axis('off')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title('原始影像')
        axes[0].axis('off')
        
        axes[1].imshow(pred_mask.cpu().numpy(), cmap=cmap)
        axes[1].set_title('預測分割')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print('🧪 測試分割評估指標...\n')
    
    # 建立測試資料
    pred = torch.randint(0, 3, (256, 256))
    target = torch.randint(0, 3, (256, 256))
    
    # 計算指標
    pixel_acc = calculate_pixel_accuracy(pred, target)
    miou = calculate_miou(pred, target, num_classes=3)
    
    print(f'像素準確率: {pixel_acc:.4f}')
    print(f'mIoU: {miou:.4f}')
    print('\n✅ 評估指標計算完成!')
