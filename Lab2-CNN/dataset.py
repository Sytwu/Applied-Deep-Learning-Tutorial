"""
資料集模組 (Dataset Module)
負責載入 CIFAR-10 資料集、定義資料前處理與資料增強流程
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Optional
from pathlib import Path


class CIFAR10Dataset:
    """
    CIFAR-10 資料集包裝器
    自動下載並載入 CIFAR-10 彩色影像資料集
    """
    
    def __init__(
        self,
        data_dir: str = './data',
        batch_size: int = 128,
        num_workers: int = 2,
        download: bool = True
    ):
        """
        初始化 CIFAR-10 資料集
        
        Args:
            data_dir: 資料儲存目錄
            batch_size: 批次大小
            num_workers: 資料載入的工作執行緒數量
            download: 是否自動下載資料集
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        
        # 確保資料目錄存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # CIFAR-10 的統計值 (ImageNet 預先計算)
        # 每個通道的平均值與標準差
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2470, 0.2435, 0.2616)
        
        # 定義訓練集的資料轉換流程 (包含資料增強)
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 隨機裁切 (保持 32x32)
            transforms.RandomHorizontalFlip(p=0.5),  # 隨機水平翻轉
            transforms.ToTensor(),  # 轉換為 Tensor (0-1)
            transforms.Normalize(self.mean, self.std)  # 正規化
        ])
        
        # 定義測試集的資料轉換流程 (不包含資料增強)
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        # 載入訓練集與測試集
        self._load_datasets()
    
    def _load_datasets(self) -> None:
        """載入訓練集與測試集"""
        print(f'📁 正在載入 CIFAR-10 資料集...')
        
        self.train_dataset = datasets.CIFAR10(
            root=str(self.data_dir),
            train=True,
            transform=self.transform_train,
            download=self.download
        )
        
        self.test_dataset = datasets.CIFAR10(
            root=str(self.data_dir),
            train=False,
            transform=self.transform_test,
            download=self.download
        )
        
        print(f'✅ CIFAR-10 資料集載入完成!')
        print(f'   訓練樣本數: {len(self.train_dataset)}')
        print(f'   測試樣本數: {len(self.test_dataset)}')
        print(f'   影像尺寸: 32x32 (RGB 彩色)')
        print(f'   類別數量: 10')
        print(f'   類別名稱: {self.get_class_names()}')
    
    def get_train_loader(self, shuffle: bool = True) -> DataLoader:
        """
        取得訓練集 DataLoader
        
        Args:
            shuffle: 是否隨機打亂資料順序
            
        Returns:
            訓練集 DataLoader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def get_test_loader(self) -> DataLoader:
        """
        取得測試集 DataLoader
        
        Returns:
            測試集 DataLoader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def get_class_names(self) -> list:
        """
        取得類別名稱列表
        
        Returns:
            類別名稱列表
        """
        return [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    def get_sample_batch(self, num_samples: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        取得一批樣本用於視覺化
        
        Args:
            num_samples: 樣本數量
            
        Returns:
            (images, labels): 影像與標籤的 Tuple
        """
        indices = torch.randint(0, len(self.test_dataset), (num_samples,))
        images, labels = [], []
        
        for idx in indices:
            img, label = self.test_dataset[idx]
            images.append(img)
            labels.append(label)
        
        return torch.stack(images), torch.tensor(labels)


def get_cifar10_dataloaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    便捷函式：直接取得訓練集與測試集的 DataLoader
    
    Args:
        data_dir: 資料儲存目錄
        batch_size: 批次大小
        num_workers: 資料載入的工作執行緒數量
        
    Returns:
        (train_loader, test_loader): 訓練集與測試集的 DataLoader
    """
    dataset = CIFAR10Dataset(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return dataset.get_train_loader(), dataset.get_test_loader()


if __name__ == '__main__':
    # 測試資料集載入
    print('🧪 測試 CIFAR-10 資料集載入...\n')
    
    dataset = CIFAR10Dataset(batch_size=32)
    train_loader = dataset.get_train_loader()
    test_loader = dataset.get_test_loader()
    
    # 顯示一個 batch 的資料形狀
    images, labels = next(iter(train_loader))
    print(f'\n📦 Batch 資料形狀:')
    print(f'   Images: {images.shape}  # (batch_size, channels, height, width)')
    print(f'   Labels: {labels.shape}  # (batch_size,)')
    print(f'\n   資料範圍: [{images.min():.3f}, {images.max():.3f}]')
    print(f'   類別範例: {[dataset.get_class_names()[labels[i]] for i in range(min(5, len(labels)))]}')
