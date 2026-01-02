"""
資料集模組 - PASCAL VOC 物件偵測
注意: 本模組為簡化版，聚焦於概念理解
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path


class VOCDetectionDataset(Dataset):
    """
    PASCAL VOC 物件偵測資料集
    
    注意: 完整的 PASCAL VOC 資料集處理需要:
    1. XML 標註檔解析
    2. 多物體處理
    3. 複雜的資料增強
    
    本實作為簡化版，建議使用 torchvision.datasets.VOCDetection
    """
    
    def __init__(self, data_dir: str, split: str = 'train'):
        """
        初始化資料集
        
        Args:
            data_dir: 資料目錄
            split: 'train' 或 'val'
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
        print(f'📁 載入 PASCAL VOC 資料集 ({split})')
        print(f'⚠️  注意: 這是簡化版實作')
        print(f'   建議使用: torchvision.datasets.VOCDetection')
        print(f'   參考: https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.VOCDetection')
    
    def __len__(self) -> int:
        return 100  # 示例數量
    
    def __getitem__(self, idx: int):
        """
        取得單個樣本
        
        Returns:
            image: 影像 Tensor
            target: 包含 boxes 和 labels 的字典
        """
        # 這裡應該實作實際的資料載入邏輯
        # 包括: 讀取影像、解析 XML、處理 Bounding Box 等
        
        # 示例回傳值
        image = torch.randn(3, 224, 224)
        target = {
            'boxes': torch.tensor([[10, 10, 100, 100]]),  # [x_min, y_min, x_max, y_max]
            'labels': torch.tensor([1])  # 類別 ID
        }
        
        return image, target


class VOCDatasetInfo:
    """PASCAL VOC 資料集資訊"""
    
    CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    @classmethod
    def get_num_classes(cls) -> int:
        return len(cls.CLASSES)
    
    @classmethod
    def get_class_name(cls, idx: int) -> str:
        return cls.CLASSES[idx] if 0 <= idx < len(cls.CLASSES) else 'unknown'


if __name__ == '__main__':
    print('📚 PASCAL VOC 資料集資訊:\n')
    print(f'類別數量: {VOCDatasetInfo.get_num_classes()}')
    print(f'類別列表: {VOCDatasetInfo.CLASSES}')
    
    print('\n💡 建議使用 torchvision 提供的實作:')
    print('   from torchvision.datasets import VOCDetection')
    print('   dataset = VOCDetection(root="./data", year="2012", image_set="train", download=True)')
