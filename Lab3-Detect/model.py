"""
模型架構模組 - 簡化版物件偵測模型
本模組實作基礎的物件偵測概念，聚焦於學習而非實際應用

注意: 完整的物件偵測系統(如 YOLO, Faster R-CNN)非常複雜
本作業旨在理解核心概念: Bounding Box, IoU, NMS
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class SimpleDetector(nn.Module):
    """
    簡化版物件偵測模型(僅用於教學)
    
    架構: Backbone (特徵提取) + Detection Head (分類 + 定位)
    """
    
    def __init__(self, num_classes: int = 20, backbone_channels: int = 512):
        """
        初始化偵測模型
        
        Args:
            num_classes: 物體類別數量 (PASCAL VOC: 20)
            backbone_channels: Backbone 輸出通道數
        """
        super(SimpleDetector, self).__init__()
        
        self.num_classes = num_classes
        
        # Backbone: 特徵提取網路
        self.backbone = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 4
            nn.Conv2d(256, backbone_channels, 3, padding=1),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(inplace=True)
        )
        
        # Detection Head: 分類 + Bounding Box 回歸
        # 輸出: (類別機率, x, y, w, h)
        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # 分類分支
        self.class_head = nn.Linear(512, num_classes)
        
        # Bounding Box 回歸分支 (x_min, y_min, x_max, y_max)
        self.bbox_head = nn.Linear(512, 4)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Returns:
            (class_scores, bbox_coords): 類別分數與邊界框座標
        """
        features = self.backbone(x)
        features = self.detection_head(features)
        
        class_scores = self.class_head(features)  # (batch, num_classes)
        bbox_coords = self.bbox_head(features)    # (batch, 4)
        
        return class_scores, bbox_coords


def create_detector(num_classes: int = 20) -> SimpleDetector:
    """創建物件偵測模型"""
    model = SimpleDetector(num_classes=num_classes)
    print(f'✅ 建立物件偵測模型 (類別數: {num_classes})')
    return model


if __name__ == '__main__':
    print('🧪 測試物件偵測模型...\n')
    model = create_detector()
    test_input = torch.randn(2, 3, 224, 224)
    class_scores, bbox_coords = model(test_input)
    print(f'輸入形狀: {test_input.shape}')
    print(f'類別分數形狀: {class_scores.shape}')
    print(f'邊界框座標形狀: {bbox_coords.shape}')
