"""
模型架構模組 (Model Architecture Module)
定義 CNN (Convolutional Neural Network) 模型用於 CIFAR-10 影像分類

學生任務:
- 實作 ConvBlock 的 __init__ 方法來建立卷積區塊
- 實作 CNN 的 __init__ 方法來組合多個卷積區塊
- 理解 CNN 的前向傳播流程
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ConvBlock(nn.Module):
    """
    卷積區塊 (Convolutional Block)
    標準組成: Conv2d → BatchNorm2d → ReLU → MaxPool2d
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool: bool = True
    ):
        """
        初始化卷積區塊
        
        Args:
            in_channels: 輸入通道數
            out_channels: 輸出通道數
            kernel_size: 卷積核大小 (預設 3x3)
            padding: Padding 大小 (預設 1)
            pool: 是否包含 MaxPooling (預設 True)
        """
        super(ConvBlock, self).__init__()
        
        # ========================================
        # TODO: 學生實作區 - 建立卷積區塊
        # ========================================
        # 建立一個卷積區塊，包含以下層（按順序）:
        #
        # 1. Conv2d: 卷積層
        #    - 使用: nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        #    - 功能: 提取空間特徵
        #    - 範例: Conv2d(3, 64, 3, padding=1) 
        #            輸入 (3, 32, 32) → 輸出 (64, 32, 32)
        #
        # 2. BatchNorm2d: 批次正規化
        #    - 使用: nn.BatchNorm2d(out_channels)
        #    - 功能: 穩定訓練，加速收斂
        #    - 注意: 正規化的通道數是輸出通道數
        #
        # 3. ReLU: 激活函數
        #    - 使用: nn.ReLU(inplace=True)
        #    - 功能: 引入非線性
        #    - inplace=True 可節省記憶體
        #
        # 4. MaxPool2d: 最大池化 (如果 pool=True)
        #    - 使用: nn.MaxPool2d(kernel_size=2, stride=2)
        #    - 功能: 降低空間維度，提取顯著特徵
        #    - 效果: (H, W) → (H/2, W/2)
        #    - 範例: (64, 32, 32) → (64, 16, 16)
        #
        # 實作方式 1 (推薦): 使用列表 + nn.Sequential
        # layers = []
        # layers.append(nn.Conv2d(...))
        # layers.append(nn.BatchNorm2d(...))
        # layers.append(nn.ReLU(...))
        # if pool:
        #     layers.append(nn.MaxPool2d(...))
        # self.block = nn.Sequential(*layers)
        #
        # 實作方式 2: 分別定義
        # self.conv = nn.Conv2d(...)
        # self.bn = nn.BatchNorm2d(...)
        # self.relu = nn.ReLU(...)
        # self.pool = nn.MaxPool2d(...) if pool else nn.Identity()
        # ========================================
        
        # TODO: 在這裡建立 self.block
        # 範例結構:
        # self.block = nn.Sequential(
        #     nn.Conv2d(...),
        #     nn.BatchNorm2d(...),
        #     nn.ReLU(...),
        #     nn.MaxPool2d(...) if pool else ...
        # )
        
        raise NotImplementedError("學生需要實作 ConvBlock.__init__")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播 (已提供，無需修改)
        
        Args:
            x: 輸入 Tensor
        
        Returns:
            處理後的 Tensor
        """
        return self.block(x)


class CNN(nn.Module):
    """
    卷積神經網路 (Convolutional Neural Network) 模型
    
    架構設計:
    輸入 (3, 32, 32) RGB 影像
      ↓
    ConvBlock 1: 3 → 64 通道, 32×32 → 16×16
      ↓
    ConvBlock 2: 64 → 128 通道, 16×16 → 8×8
      ↓
    ConvBlock 3: 128 → 256 通道, 8×8 → 4×4
      ↓
    ConvBlock 4: 256 → 512 通道, 4×4 → 2×2
      ↓
    Global Average Pooling: (512, 2, 2) → (512, 1, 1)
      ↓
    Flatten: (512, 1, 1) → (512,)
      ↓
    Dropout + Linear: (512,) → (10,)
    """
    
    def __init__(
        self,
        in_channels: int = 3,  # RGB 影像
        num_classes: int = 10,  # CIFAR-10 有 10 個類別
        base_channels: int = 64,
        dropout_rate: float = 0.5
    ):
        """
        初始化 CNN 模型
        
        Args:
            in_channels: 輸入通道數 (RGB=3, 灰階=1)
            num_classes: 輸出類別數量
            base_channels: 基礎通道數 (第一層的輸出通道)
            dropout_rate: Dropout 比率
        """
        super(CNN, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # ========================================
        # TODO: 學生實作區 - 建立特徵提取器
        # ========================================
        # 使用 nn.Sequential 組合 4 個 ConvBlock
        #
        # 通道數變化規律: 每層通道數翻倍
        # Block 1: in_channels (3) → base_channels (64)
        # Block 2: base_channels (64) → base_channels * 2 (128)
        # Block 3: base_channels * 2 (128) → base_channels * 4 (256)
        # Block 4: base_channels * 4 (256) → base_channels * 8 (512)
        #
        # 空間尺寸變化 (因為 MaxPool):
        # Block 1: 32×32 → 16×16
        # Block 2: 16×16 → 8×8
        # Block 3: 8×8 → 4×4
        # Block 4: 4×4 → 2×2
        #
        # 實作範例:
        # self.features = nn.Sequential(
        #     ConvBlock(in_channels, base_channels, pool=True),
        #     ConvBlock(base_channels, base_channels * 2, pool=True),
        #     ConvBlock(...),  # 你來填
        #     ConvBlock(...)   # 你來填
        # )
        # ========================================
        
        # TODO: 建立 self.features (特徵提取器)
        self.features = None  # 替換為你的實作
        
        # ========================================
        # TODO: 學生實作區 - 建立全域平均池化
        # ========================================
        # Global Average Pooling (GAP):
        # - 將 (batch, 512, 2, 2) 轉為 (batch, 512, 1, 1)
        # - 對每個 channel 的整個空間維度取平均
        # - 使用: nn.AdaptiveAvgPool2d((1, 1))
        # - 優勢: 參數更少，對輸入尺寸更靈活
        # ========================================
        
        # TODO: 建立 self.global_avg_pool
        self.global_avg_pool = None  # 替換為你的實作
        
        # ========================================
        # TODO: 學生實作區 - 建立分類器
        # ========================================
        # 分類器包含:
        # 1. Dropout: nn.Dropout(dropout_rate)
        #    - 防止過擬合
        #    - 訓練時隨機丟棄 50% 的神經元
        #
        # 2. Linear: nn.Linear(base_channels * 8, num_classes)
        #    - 最終的分類層
        #    - 輸入: 512 維特徵向量
        #    - 輸出: 10 個類別的分數
        #
        # 使用 nn.Sequential 組合:
        # self.classifier = nn.Sequential(
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(base_channels * 8, num_classes)
        # )
        # ========================================
        
        # TODO: 建立 self.classifier
        self.classifier = None  # 替換為你的實作
        
        # 檢查是否完成實作
        if self.features is None or self.global_avg_pool is None or self.classifier is None:
            raise NotImplementedError("學生需要完成 CNN.__init__ 的實作")
        
        # 初始化權重 (已提供)
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """
        初始化模型權重 (已提供，無需修改)
        使用 Kaiming 初始化，適合 ReLU 激活函數
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播 (已提供，無需修改)
        
        Args:
            x: 輸入 Tensor，形狀為 (batch_size, 3, 32, 32)
            
        Returns:
            輸出 Tensor，形狀為 (batch_size, num_classes)
        
        Tensor 形狀變化:
            輸入:           (batch_size, 3, 32, 32)
            Conv Block 1:   (batch_size, 64, 16, 16)
            Conv Block 2:   (batch_size, 128, 8, 8)
            Conv Block 3:   (batch_size, 256, 4, 4)
            Conv Block 4:   (batch_size, 512, 2, 2)
            Global Avg Pool: (batch_size, 512, 1, 1)
            Flatten:        (batch_size, 512)
            Classifier:     (batch_size, 10)
        """
        # 特徵提取
        x = self.features(x)  # (batch, 3, 32, 32) → (batch, 512, 2, 2)
        
        # 全域平均池化
        x = self.global_avg_pool(x)  # (batch, 512, 2, 2) → (batch, 512, 1, 1)
        
        # 展平
        x = torch.flatten(x, 1)  # (batch, 512, 1, 1) → (batch, 512)
        
        # 分類
        x = self.classifier(x)  # (batch, 512) → (batch, 10)
        
        return x
    
    def get_num_parameters(self) -> int:
        """
        計算模型總參數量
        
        Returns:
            模型總參數數量
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_cnn_model(
    in_channels: int = 3,
    num_classes: int = 10,
    base_channels: int = 64,
    dropout_rate: float = 0.5
) -> CNN:
    """
    便捷函式：創建 CNN 模型
    
    Args:
        in_channels: 輸入通道數
        num_classes: 輸出類別數量
        base_channels: 基礎通道數
        dropout_rate: Dropout 比率
        
    Returns:
        CNN 模型實例
    """
    model = CNN(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        dropout_rate=dropout_rate
    )
    
    print(f'✅ 建立 CNN 模型成功!')
    print(f'   基礎通道數: {base_channels}')
    print(f'   總參數量: {model.get_num_parameters():,}')
    
    return model


if __name__ == '__main__':
    # 測試模型
    print('🧪 測試 CNN 模型...\n')
    
    try:
        model = create_cnn_model()
        
        # 建立測試輸入 (CIFAR-10 格式: RGB 32x32)
        batch_size = 4
        test_input = torch.randn(batch_size, 3, 32, 32)
        
        # 前向傳播
        output = model(test_input)
        
        print(f'\n📊 模型測試結果:')
        print(f'   輸入形狀: {test_input.shape}')
        print(f'   輸出形狀: {output.shape}')
        print(f'   預期輸出形狀: ({batch_size}, 10)')
        
        # 驗證輸出形狀
        assert output.shape == (batch_size, 10), '輸出形狀不正確!'
        print(f'\n✅ 模型測試通過!')
        
    except NotImplementedError as e:
        print(f'\n⚠️  {e}')
        print('\n請完成以下 TODO 區塊:')
        print('1. ConvBlock.__init__() - 建立卷積區塊')
        print('2. CNN.__init__() - 建立特徵提取器、全域平均池化、分類器')
