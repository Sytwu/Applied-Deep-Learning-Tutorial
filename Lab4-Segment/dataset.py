"""
資料集模組 - Oxford-IIIT Pet 語義分割
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path


class OxfordPetDataset(Dataset):
    """
    Oxford-IIIT Pet 資料集用於語義分割
    
    注意: 建議使用 torchvision.datasets.OxfordIIITPet
    """
    
    def __init__(self, data_dir: str, split: str = 'trainval', target_type: str = 'segmentation'):
        """
        初始化資料集
        
        Args:
            data_dir: 資料目錄
            split: 'trainval' 或 'test'
            target_type: 'segmentation' 或 'category'
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.target_type = target_type
        
        print(f'📁 載入 Oxford-IIIT Pet 資料集 ({split})')
        print(f'⚠️  注意: 這是簡化版實作')
        print(f'   建議使用: torchvision.datasets.OxfordIIITPet')
        print(f'   參考: https://pytorch.org/vision/stable/datasets.html#oxford-iiit-pet')
    
    def __len__(self) -> int:
        return 100  # 示例數量
    
    def __getitem__(self, idx: int):
        """
        取得單個樣本
        
        Returns:
            image: 影像 Tensor (3, H, W)
            mask: 分割 mask Tensor (H, W)
        """
        # 這裡應該實作實際的資料載入邏輯
        # 包括: 讀取影像、讀取分割 mask、資料增強等
        
        # 示例回傳值
        image = torch.randn(3, 256, 256)
        mask = torch.randint(0, 3, (256, 256))  # 3 classes: foreground/background/border
        
        return image, mask


class PetDatasetInfo:
    """Oxford-IIIT Pet 資料集資訊"""
    
    CLASSES = ['foreground', 'background', 'border']
    
    @classmethod
    def get_num_classes(cls) -> int:
        return len(cls.CLASSES)
    
    @classmethod
    def get_class_name(cls, idx: int) -> str:
        return cls.CLASSES[idx] if 0 <= idx < len(cls.CLASSES) else 'unknown'


if __name__ == '__main__':
    print('📚 Oxford-IIIT Pet 資料集資訊:\n')
    print(f'類別數量: {PetDatasetInfo.get_num_classes()}')
    print(f'類別列表: {PetDatasetInfo.CLASSES}')
    
    print('\n💡 建議使用 torchvision 提供的實作:')
    print('   from torchvision.datasets import OxfordIIITPet')
    print('   dataset = OxfordIIITPet(root="./data", split="trainval", target_types="segmentation", download=True)')
