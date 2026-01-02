"""
資料集模組 (Dataset Module)
處理從 JSON 格式載入 MNIST 資料集並建立 DataLoader

學生任務:
- 實作 MNISTDataset 類別來從 JSON 載入影像與標籤
- 實作 __getitem__ 來回傳轉換後的影像與標籤
- 理解 DataLoader 如何與自訂資料集搭配運作
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
from typing import Tuple, Optional


class MNISTDataset(Dataset):
    """
    自訂 MNIST 資料集，從 JSON 標註檔載入
    
    JSON 格式:
    {
        "samples": [
            {
                "id": "00001",
                "image_path": "train/00001.png",
                "label": 5
            },
            ...
        ]
    }
    """
    
    def __init__(self, json_path: str, data_root: str, transform=None):
        """
        初始化 MNIST 資料集
        
        Args:
            json_path (str): JSON 標註檔路徑 (例如: 'data/train.json')
            data_root (str): 包含影像檔案的根目錄
            transform: 要套用到影像的 torchvision transforms
        """
        self.data_root = Path(data_root)
        self.transform = transform
        
        # ========================================
        # TODO: 學生實作區
        # ========================================
        # 載入 JSON 檔案並提取樣本列表
        #
        # 步驟:
        # 1. 使用 json.load() 開啟並讀取 JSON 檔案
        # 2. 從 JSON 資料中提取 'samples' 列表
        # 3. 儲存到 self.samples
        #
        # 範例:
        # with open(json_path, 'r') as f:
        #     data = json.load(f)
        # self.samples = data['samples']
        # ========================================
        
        self.samples = []  # TODO: 從 JSON 檔案載入
        
        print(f'從 {json_path} 載入了 {len(self.samples)} 個樣本')
    
    def __len__(self) -> int:
        """
        回傳資料集中的樣本總數
        
        Returns:
            int: 資料集中的樣本數量
        """
        # ========================================
        # TODO: 學生實作區
        # ========================================
        # 回傳 self.samples 的長度
        # 這會告訴 DataLoader 資料集中有多少個樣本
        # ========================================
        
        return 0  # TODO: 回傳實際長度
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        從資料集中取得單一樣本
        
        Args:
            idx (int): 要取得的樣本索引
        
        Returns:
            tuple: (影像, 標籤) 其中
                   影像是形狀為 (1, 28, 28) 的 torch.Tensor
                   標籤是 0-9 的整數
        """
        # ========================================
        # TODO: 學生實作區
        # ========================================
        # 實作資料載入邏輯:
        #
        # 步驟 1: 取得樣本資訊
        #   sample = self.samples[idx]
        #   image_path = sample['image_path']
        #   label = sample['label']
        #
        # 步驟 2: 載入影像
        #   - 建構完整路徑: self.data_root / image_path
        #   - 使用 PIL 開啟影像: Image.open(full_path)
        #   - 如果需要，轉換為灰階: image.convert('L')
        #
        # 步驟 3: 套用轉換
        #   - 如果 self.transform 不是 None，套用它: image = self.transform(image)
        #
        # 步驟 4: 回傳影像與標籤
        #   - return image, label
        #
        # 小提示:
        # - PIL 影像需要轉換為張量 (transforms.ToTensor() 會處理這件事)
        # - 確保正確處理 Path 的連接
        # - 標籤應該是整數 (0-9)
        # ========================================
        
        raise NotImplementedError("學生需要實作 __getitem__")


def get_transforms(train: bool = True):
    """
    取得訓練或測試用的資料轉換
    
    Args:
        train (bool): 是否使用訓練用轉換
    
    Returns:
        transforms.Compose: 組合的轉換
    """
    if train:
        # 訓練用轉換 (可以在這裡加入資料增強)
        return transforms.Compose([
            transforms.ToTensor(),  # 將 PIL Image 轉為 Tensor (0-1)
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的平均值與標準差
        ])
    else:
        # 測試用轉換 (不做增強)
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])


def create_dataloaders(
    train_json: str = './data/train.json',
    test_json: str = './data/test.json',
    data_root: str = './data',
    batch_size: int = 64,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    建立訓練與測試用的 DataLoaders
    
    Args:
        train_json: 訓練 JSON 檔路徑
        test_json: 測試 JSON 檔路徑
        data_root: 包含影像的根目錄
        batch_size: DataLoader 的批次大小
        num_workers: 資料載入的工作執行緒數量
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # ========================================
    # TODO: 學生實作區
    # ========================================
    # 建立訓練與測試資料集，然後包裝成 DataLoaders
    #
    # 步驟 1: 建立資料集
    #   train_dataset = MNISTDataset(
    #       json_path=train_json,
    #       data_root=data_root,
    #       transform=get_transforms(train=True)
    #   )
    #   test_dataset = MNISTDataset(
    #       json_path=test_json,
    #       data_root=data_root,
    #       transform=get_transforms(train=False)
    #   )
    #
    # 步驟 2: 建立 DataLoaders
    #   train_loader = DataLoader(
    #       train_dataset,
    #       batch_size=batch_size,
    #       shuffle=True,  # 打亂訓練資料
    #       num_workers=num_workers,
    #       pin_memory=torch.cuda.is_available()  # 加速 GPU 傳輸
    #   )
    #   test_loader = DataLoader(
    #       test_dataset,
    #       batch_size=batch_size,
    #       shuffle=False,  # 不打亂測試資料
    #       num_workers=num_workers,
    #       pin_memory=torch.cuda.is_available()
    #   )
    #
    # 步驟 3: 回傳兩個 loaders
    #   return train_loader, test_loader
    #
    # 小提示:
    # - shuffle=True 對訓練有幫助，讓模型更好地泛化
    # - shuffle=False 對測試確保評估一致性
    # - pin_memory=True 加速 CPU 到 GPU 的資料傳輸
    # ========================================
    
    raise NotImplementedError("學生需要實作 create_dataloaders")


if __name__ == '__main__':
    # 測試資料集與 dataloader
    print('🧪 測試 MNIST 資料集...\n')
    
    try:
        # 建立 dataloaders
        train_loader, test_loader = create_dataloaders(
            batch_size=32,
            num_workers=0  # 測試時使用 0 以避免多執行緒問題
        )
        
        # 測試載入一個批次
        images, labels = next(iter(train_loader))
        
        print(f'✅ DataLoader 測試成功!')
        print(f'   批次形狀: {images.shape}  # (batch_size, channels, height, width)')
        print(f'   標籤形狀: {labels.shape}  # (batch_size,)')
        print(f'   數值範圍: [{images.min():.3f}, {images.max():.3f}]')
        print(f'   樣本標籤: {labels[:5].tolist()}')
        
    except (FileNotFoundError, NotImplementedError) as e:
        print(f'⚠️  資料集尚未準備好: {e}')
        print('\n要完成這個作業:')
        print('1. 在 ./data/ 中準備 train.json 和 test.json')
        print('2. 實作 MNISTDataset.__getitem__')
        print('3. 實作 create_dataloaders')
