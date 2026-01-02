"""
訓練器模組 (Trainer Module)
封裝訓練與驗證流程的 Trainer 類別
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from pathlib import Path

from utils import AverageMeter, save_checkpoint


class Trainer:
    """
    訓練器類別
    負責模型的訓練、驗證、學習率調度與模型儲存
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        save_dir: str = './checkpoints',
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ):
        """
        初始化訓練器
        
        Args:
            model: 要訓練的模型
            train_loader: 訓練集 DataLoader
            val_loader: 驗證集 DataLoader
            criterion: 損失函數
            optimizer: 優化器
            device: 運算裝置
            save_dir: 模型儲存目錄
            scheduler: 學習率調度器 (可選)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 記錄訓練歷史
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # 最佳指標
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        訓練一個 epoch
        
        Args:
            epoch: 當前 epoch 編號
            
        Returns:
            (平均損失, 平均準確率)
        """
        self.model.train()  # 設定為訓練模式
        
        losses = AverageMeter()
        accs = AverageMeter()
        
        # 使用 tqdm 顯示訓練進度
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for images, labels in pbar:
            # 將資料移至指定裝置
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            batch_size = images.size(0)
            
            # TODO: Student Implementation
            # 請完成以下訓練步驟:
            # 1. 清空梯度
            # 2. 前向傳播
            # 3. 計算損失
            # 4. 反向傳播
            # 5. 更新權重
            
            # 提示: 使用 optimizer.zero_grad(), model(), criterion(), loss.backward(), optimizer.step()
            
            # 1. 清空梯度
            self.optimizer.zero_grad()
            
            # 2. 前向傳播
            outputs = self.model(images)
            
            # 3. 計算損失
            loss = self.criterion(outputs, labels)
            
            # 4. 反向傳播
            loss.backward()
            
            # 5. 更新權重
            self.optimizer.step()
            
            # 計算準確率
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            acc = 100.0 * correct / batch_size
            
            # 更新統計
            losses.update(loss.item(), batch_size)
            accs.update(acc, batch_size)
            
            # 更新進度條顯示
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{accs.avg:.2f}%'
            })
        
        return losses.avg, accs.avg
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        在驗證集上評估模型
        
        Args:
            epoch: 當前 epoch 編號
            
        Returns:
            (平均損失, 平均準確率)
        """
        self.model.eval()  # 設定為評估模式
        
        losses = AverageMeter()
        accs = AverageMeter()
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():  # 驗證時不需要計算梯度
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                batch_size = images.size(0)
                
                # 前向傳播
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # 計算準確率
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                acc = 100.0 * correct / batch_size
                
                # 更新統計
                losses.update(loss.item(), batch_size)
                accs.update(acc, batch_size)
                
                # 更新進度條顯示
                pbar.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Acc': f'{accs.avg:.2f}%'
                })
        
        return losses.avg, accs.avg
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        執行完整訓練流程
        
        Args:
            num_epochs: 訓練輪數
            
        Returns:
            包含訓練歷史的字典
        """
        print(f'\n🚀 開始訓練，共 {num_epochs} 個 Epochs\n')
        print('=' * 70)
        
        for epoch in range(1, num_epochs + 1):
            # 訓練一個 epoch
            train_loss, train_acc = self.train_one_epoch(epoch)
            
            # 驗證
            val_loss, val_acc = self.validate(epoch)
            
            # 記錄歷史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # 學習率調度
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 顯示 epoch 總結
            print(f'\nEpoch {epoch}/{num_epochs} Summary:')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            # 儲存最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                print(f'  🎉 新的最佳驗證準確率: {val_acc:.2f}%')
            
            # 儲存檢查點
            save_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                best_metric=self.best_val_acc,
                save_path=str(save_path),
                is_best=is_best
            )
            
            print('=' * 70)
        
        print(f'\n✅ 訓練完成!')
        print(f'   最佳驗證準確率: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }
