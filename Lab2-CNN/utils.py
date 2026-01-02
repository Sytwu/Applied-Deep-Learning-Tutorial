"""
工具函式模組 (Utility Functions)
提供訓練過程中常用的輔助函式，包含隨機種子設定、裝置偵測、模型儲存與載入、訓練曲線繪製等。
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import json


def set_seed(seed: int = 42) -> None:
    """
    設定所有隨機種子以確保實驗可重現性
    
    Args:
        seed: 隨機種子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 以下設定可能會降低效能，但可提高可重現性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    自動偵測可用的運算裝置 (CUDA > MPS > CPU)
    
    Returns:
        torch.device: 可用的運算裝置
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'🚀 使用 CUDA 裝置: {torch.cuda.get_device_name(0)}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('🍎 使用 Apple Silicon MPS 裝置')
    else:
        device = torch.device('cpu')
        print('💻 使用 CPU 裝置')
    
    return device


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    save_path: str,
    is_best: bool = False
) -> None:
    """
    儲存模型檢查點
    
    Args:
        model: 要儲存的模型
        optimizer: 優化器
        epoch: 當前訓練輪數
        best_metric: 最佳指標值
        save_path: 儲存路徑
        is_best: 是否為最佳模型
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
    }
    
    # 確保目錄存在
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = str(Path(save_path).parent / 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f'✅ 儲存最佳模型至 {best_path}')


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    device: torch.device
) -> Dict:
    """
    載入模型檢查點
    
    Args:
        model: 要載入權重的模型
        optimizer: 優化器 (可選)
        checkpoint_path: 檢查點路徑
        device: 運算裝置
        
    Returns:
        包含 epoch 和 best_metric 的字典
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f'找不到檢查點檔案: {checkpoint_path}')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f'✅ 成功載入檢查點: Epoch {checkpoint["epoch"]}, Best Metric: {checkpoint["best_metric"]:.4f}')
    
    return {
        'epoch': checkpoint['epoch'],
        'best_metric': checkpoint['best_metric']
    }


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None
) -> None:
    """
    繪製訓練曲線 (損失與準確率)
    
    Args:
        train_losses: 訓練損失列表
        val_losses: 驗證損失列表
        train_accs: 訓練準確率列表
        val_accs: 驗證準確率列表
        save_path: 儲存圖表的路徑 (可選)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 繪製損失曲線
    ax1.plot(epochs, train_losses, 'b-', label='訓練損失', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='驗證損失', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('訓練與驗證損失曲線', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 繪製準確率曲線
    ax2.plot(epochs, train_accs, 'b-', label='訓練準確率', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='驗證準確率', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('訓練與驗證準確率曲線', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'📊 訓練曲線已儲存至 {save_path}')
    
    plt.show()


def save_metrics(metrics: Dict, save_path: str) -> None:
    """
    儲存訓練指標至 JSON 檔案
    
    Args:
        metrics: 包含訓練指標的字典
        save_path: 儲存路徑
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    
    print(f'📝 訓練指標已儲存至 {save_path}')


class AverageMeter:
    """計算並儲存平均值與當前值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        更新統計值
        
        Args:
            val: 數值
            n: 樣本數量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
