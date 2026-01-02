"""
模型架構模組 - U-Net 語義分割模型
實作經典的 U-Net 架構用於像素級別的影像分割
"""

import torch
import torch.nn as nn
from typing import List


class DoubleConv(nn.Module):
    """雙卷積區塊: Conv → BN → ReLU → Conv → BN → ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    U-Net 架構
    
    Encoder (下採樣路徑):
        提取高階特徵，逐步降低解析度
    
    Decoder (上採樣路徑):
        恢復空間解析度，產生分割 mask
    
    Skip Connections:
        連接 Encoder 和 Decoder 對應層，保留細節資訊
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 3, base_channels: int = 64):
        """
        初始化 U-Net
        
        Args:
            in_channels: 輸入通道數 (RGB=3)
            num_classes: 輸出類別數 (Oxford Pet: 3 = foreground/background/border)
            base_channels: 基礎通道數
        """
        super(UNet, self).__init__()
        
        # TODO: Student Implementation
        # 請觀察 U-Net 的架構設計:
        # 1. Encoder 部分如何逐步降低解析度
        # 2. Decoder 部分如何逐步恢復解析度
        # 3. Skip Connections 如何連接對應的層
        
        # Encoder (下採樣路徑)
        self.enc1 = DoubleConv(in_channels, base_channels)              # 64
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)        # 128
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)    # 256
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)    # 512
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck (最底層)
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)  # 1024
        
        # Decoder (上採樣路徑)
        self.upconv4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)  # 拼接後: 1024 → 512
        
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)   # 拼接後: 512 → 256
        
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)   # 拼接後: 256 → 128
        
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)        # 拼接後: 128 → 64
        
        # 輸出層: 1x1 卷積產生分割 mask
        self.out_conv = nn.Conv2d(base_channels, num_classes, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Tensor 形狀變化 (假設輸入 256x256):
            輸入: (batch, 3, 256, 256)
            
            Encoder:
                enc1: (batch, 64, 256, 256)
                pool1 → enc2: (batch, 128, 128, 128)
                pool2 → enc3: (batch, 256, 64, 64)
                pool3 → enc4: (batch, 512, 32, 32)
                pool4 → bottleneck: (batch, 1024, 16, 16)
            
            Decoder:
                upconv4 + skip4 → dec4: (batch, 512, 32, 32)
                upconv3 + skip3 → dec3: (batch, 256, 64, 64)
                upconv2 + skip2 → dec2: (batch, 128, 128, 128)
                upconv1 + skip1 → dec1: (batch, 64, 256, 256)
            
            輸出: (batch, num_classes, 256, 256)
        """
        # Encoder 前向傳播 (同時儲存 skip connections)
        enc1 = self.enc1(x)        # 64, 256, 256
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)        # 128, 128, 128
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)        # 256, 64, 64
        x = self.pool3(enc3)
        
        enc4 = self.enc4(x)        # 512, 32, 32
        x = self.pool4(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)     # 1024, 16, 16
        
        # Decoder 前向傳播 (使用 skip connections)
        x = self.upconv4(x)        # 512, 32, 32
        x = torch.cat([x, enc4], dim=1)  # 拼接 skip connection
        x = self.dec4(x)           # 512, 32, 32
        
        x = self.upconv3(x)        # 256, 64, 64
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)           # 256, 64, 64
        
        x = self.upconv2(x)        # 128, 128, 128
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)           # 128, 128, 128
        
        x = self.upconv1(x)        # 64, 256, 256
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)           # 64, 256, 256
        
        # 輸出分割 mask
        x = self.out_conv(x)       # num_classes, 256, 256
        
        return x
    
    def get_num_parameters(self) -> int:
        """計算模型總參數量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_unet(in_channels: int = 3, num_classes: int = 3, base_channels: int = 64) -> UNet:
    """創建 U-Net 模型"""
    model = UNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels)
    print(f'✅ 建立 U-Net 模型')
    print(f'   輸入通道: {in_channels}')
    print(f'   輸出類別: {num_classes}')
    print(f'   總參數量: {model.get_num_parameters():,}')
    return model


if __name__ == '__main__':
    print('🧪 測試 U-Net 模型...\n')
    
    model = create_unet()
    test_input = torch.randn(2, 3, 256, 256)
    output = model(test_input)
    
    print(f'\n📊 模型測試結果:')
    print(f'   輸入形狀: {test_input.shape}')
    print(f'   輸出形狀: {output.shape}')
    print(f'   預期輸出形狀: (2, 3, 256, 256)')
    
    assert output.shape == (2, 3, 256, 256), '輸出形狀不正確!'
    print(f'\n✅ 模型測試通過!')
