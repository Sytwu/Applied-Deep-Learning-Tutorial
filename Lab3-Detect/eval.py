"""
評估模組 - 物件偵測評估指標
包含 IoU, NMS, mAP 等核心函式
"""

import torch
import numpy as np
from typing import List, Tuple


def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    計算兩個邊界框的 IoU (Intersection over Union)
    
    Args:
        box1: 邊界框 1, 格式 [x_min, y_min, x_max, y_max]
        box2: 邊界框 2, 格式 [x_min, y_min, x_max, y_max]
    
    Returns:
        IoU 值, 範圍 [0, 1]
    """
    # TODO: Student Implementation
    # 請完成 IoU 計算
    # 提示:
    # 1. 計算交集區域的座標
    # 2. 計算交集面積
    # 3. 計算聯集面積 = 面積1 + 面積2 - 交集面積
    # 4. IoU = 交集面積 / 聯集面積
    
    # 計算交集區域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 計算交集面積
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 計算各邊界框面積
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 計算聯集面積
    union = area1 + area2 - intersection
    
    # 計算 IoU
    iou = intersection / union if union > 0 else 0
    
    return iou


def non_maximum_suppression(
    boxes: List[torch.Tensor],
    scores: List[float],
    iou_threshold: float = 0.5
) -> List[int]:
    """
    非極大值抑制 (Non-Maximum Suppression)
    移除重複的偵測框，保留信心度最高的
    
    Args:
        boxes: 邊界框列表
        scores: 對應的信心度分數
        iou_threshold: IoU 閾值
    
    Returns:
        保留的邊界框索引列表
    """
    # TODO: Student Implementation  
    # NMS 演算法步驟:
    # 1. 依照 scores 降序排列
    # 2. 取出分數最高的框
    # 3. 移除與其 IoU > threshold 的其他框
    # 4. 重複步驟 2-3 直到沒有框剩餘
    
    if len(boxes) == 0:
        return []
    
    # 簡化版實作(供參考)
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        indices = indices[1:]
        
        # 移除與當前框 IoU 過高的框
        indices = [
            i for i in indices
            if calculate_iou(boxes[current], boxes[i]) < iou_threshold
        ]
    
    return keep


def calculate_map(predictions: List, ground_truths: List, iou_threshold: float = 0.5) -> float:
    """
    計算 mAP (mean Average Precision)
    
    注意: 這是簡化版實作，完整版需考慮多類別、多物體等情況
    
    Args:
        predictions: 預測結果列表
        ground_truths: 真實標註列表
        iou_threshold: IoU 閾值
    
    Returns:
        mAP 值
    """
    # TODO: Student Implementation
    # 完整的 mAP 計算較為複雜，建議參考:
    # https://github.com/rafaelpadilla/Object-Detection-Metrics
    
    # 這裡提供簡化的概念性實作
    print("⚠️  mAP 計算需要大量樣本，建議參考完整實作")
    return 0.0


if __name__ == '__main__':
    print('🧪 測試 IoU 計算...\n')
    
    # 測試 IoU
    box1 = torch.tensor([0, 0, 10, 10])
    box2 = torch.tensor([5, 5, 15, 15])
    iou = calculate_iou(box1, box2)
    print(f'Box 1: {box1.tolist()}')
    print(f'Box 2: {box2.tolist()}')
    print(f'IoU: {iou:.4f}')
    
    # 測試 NMS
    print('\n🧪 測試 NMS...\n')
    boxes = [
        torch.tensor([0, 0, 10, 10]),
        torch.tensor([1, 1, 11, 11]),  # 與第一個框重疊
        torch.tensor([20, 20, 30, 30])  # 不重疊
    ]
    scores = [0.9, 0.8, 0.95]
    keep_indices = non_maximum_suppression(boxes, scores, iou_threshold=0.5)
    print(f'保留的框索引: {keep_indices}')
