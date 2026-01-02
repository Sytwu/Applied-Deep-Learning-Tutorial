"""
評估模組 (Evaluation Module)
提供模型評估指標計算與預測結果視覺化功能
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from typing import Tuple, List, Optional
from pathlib import Path
from tqdm import tqdm


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    評估模型在給定資料集上的表現
    
    Args:
        model: 要評估的模型
        data_loader: 資料載入器
        device: 運算裝置
        criterion: 損失函數 (可選)
        
    Returns:
        (損失, 準確率, 所有預測, 所有標籤)
    """
    model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='評估中'):
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向傳播
            outputs = model(images)
            
            # 計算損失
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
            
            # 取得預測結果
            _, predicted = torch.max(outputs, 1)
            
            # 收集預測與標籤
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 計算平均損失與準確率
    avg_loss = total_loss / len(data_loader.dataset) if criterion else 0.0
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = 100.0 * (all_predictions == all_labels).sum() / len(all_labels)
    
    return avg_loss, accuracy, all_predictions, all_labels


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    繪製混淆矩陣 (Confusion Matrix)
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        class_names: 類別名稱列表
        save_path: 儲存路徑 (可選)
        figsize: 圖表尺寸
    """
    # 計算混淆矩陣
    cm = confusion_matrix(y_true, y_pred)
    
    # 繪製熱力圖
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': '樣本數量'}
    )
    
    plt.title('混淆矩陣 (Confusion Matrix)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('預測標籤', fontsize=12)
    plt.ylabel('真實標籤', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'📊 混淆矩陣已儲存至 {save_path}')
    
    plt.show()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> None:
    """
    輸出分類報告 (Precision, Recall, F1-Score)
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        class_names: 類別名稱列表
    """
    print('\n' + '=' * 70)
    print('分類報告 (Classification Report)')
    print('=' * 70)
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    print(report)


def visualize_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    num_samples: int = 16,
    save_path: Optional[str] = None
) -> None:
    """
    視覺化模型預測結果
    
    Args:
        model: 訓練好的模型
        data_loader: 資料載入器
        device: 運算裝置
        class_names: 類別名稱列表
        num_samples: 要顯示的樣本數量
        save_path: 儲存路徑 (可選)
    """
    model.eval()
    
    # 取得一批資料
    images, labels = next(iter(data_loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    # 進行預測
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)
    
    # 將資料移回 CPU
    images = images.cpu()
    predictions = predictions.cpu().numpy()
    confidences = confidences.cpu().numpy()
    labels = labels.numpy()
    
    # 繪製結果
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx in range(num_samples):
        ax = axes[idx]
        
        # 顯示影像 (去除正規化)
        img = images[idx].squeeze()
        mean, std = 0.1307, 0.3081
        img = img * std + mean  # 反正規化
        img = torch.clamp(img, 0, 1)
        
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        # 設定標題 (正確: 綠色，錯誤: 紅色)
        true_label = class_names[labels[idx]]
        pred_label = class_names[predictions[idx]]
        confidence = confidences[idx] * 100
        
        is_correct = labels[idx] == predictions[idx]
        color = 'green' if is_correct else 'red'
        
        title = f'真實: {true_label}\n預測: {pred_label} ({confidence:.1f}%)'
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
    
    # 隱藏多餘的子圖
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('模型預測結果視覺化', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'🖼️  預測視覺化已儲存至 {save_path}')
    
    plt.show()


def visualize_misclassified(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    num_samples: int = 16,
    save_path: Optional[str] = None
) -> None:
    """
    視覺化模型預測錯誤的樣本
    
    Args:
        model: 訓練好的模型
        data_loader: 資料載入器
        device: 運算裝置
        class_names: 類別名稱列表
        num_samples: 要顯示的樣本數量
        save_path: 儲存路徑 (可選)
    """
    model.eval()
    
    misclassified_images = []
    misclassified_preds = []
    misclassified_labels = []
    misclassified_confs = []
    
    # 收集預測錯誤的樣本
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
            
            # 找出預測錯誤的樣本
            incorrect_mask = predictions != labels
            
            if incorrect_mask.sum() > 0:
                misclassified_images.append(images[incorrect_mask].cpu())
                misclassified_preds.append(predictions[incorrect_mask].cpu())
                misclassified_labels.append(labels[incorrect_mask].cpu())
                misclassified_confs.append(confidences[incorrect_mask].cpu())
            
            # 收集足夠的樣本後停止
            if sum(img.size(0) for img in misclassified_images) >= num_samples:
                break
    
    if len(misclassified_images) == 0:
        print('🎉 沒有找到預測錯誤的樣本!')
        return
    
    # 合併樣本
    misclassified_images = torch.cat(misclassified_images)[:num_samples]
    misclassified_preds = torch.cat(misclassified_preds)[:num_samples]
    misclassified_labels = torch.cat(misclassified_labels)[:num_samples]
    misclassified_confs = torch.cat(misclassified_confs)[:num_samples]
    
    # 繪製結果
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx in range(min(num_samples, len(misclassified_images))):
        ax = axes[idx]
        
        # 顯示影像
        img = misclassified_images[idx].squeeze()
        mean, std = 0.1307, 0.3081
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        # 設定標題
        true_label = class_names[misclassified_labels[idx]]
        pred_label = class_names[misclassified_preds[idx]]
        confidence = misclassified_confs[idx] * 100
        
        title = f'真實: {true_label}\n預測: {pred_label} ({confidence:.1f}%)'
        ax.set_title(title, fontsize=10, color='red', fontweight='bold')
    
    # 隱藏多餘的子圖
    for idx in range(len(misclassified_images), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('預測錯誤的樣本', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'🖼️  錯誤案例視覺化已儲存至 {save_path}')
    
    plt.show()


if __name__ == '__main__':
    print('✅ 評估模組載入成功!')
    print('   可用函式:')
    print('   - evaluate_model(): 評估模型表現')
    print('   - plot_confusion_matrix(): 繪製混淆矩陣')
    print('   - print_classification_report(): 輸出分類報告')
    print('   - visualize_predictions(): 視覺化預測結果')
    print('   -visualize_misclassified(): 視覺化預測錯誤的樣本')
