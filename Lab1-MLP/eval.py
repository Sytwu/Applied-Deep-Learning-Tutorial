"""
評估模組 (Evaluation Module)
提供模型評估指標與預測結果視覺化的輔助函式

注意: 本模組的函式已完整實作，學生可直接使用
學生的主要實作重點在 train.py 的訓練與驗證迴圈
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


def calculate_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    計算分類準確率
    
    Args:
        predictions (np.ndarray): 預測的類別標籤，形狀 (N,)
        labels (np.ndarray): 真實的類別標籤，形狀 (N,)
    
    Returns:
        float: 準確率百分比 (0-100)
    """
    correct = (predictions == labels).sum()
    total = len(labels)
    accuracy = 100.0 * correct / total
    return accuracy


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    在資料集上評估模型
    
    Args:
        model: 要評估的模型
        data_loader: 資料集的 DataLoader
        device: 運算裝置 (cuda/mps/cpu)
        criterion: 損失函數 (可選)
        
    Returns:
        tuple: (損失, 準確率, 所有預測, 所有標籤)
    """
    model.eval()  # 設定模型為評估模式
    
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
            
            # 取得預測
            _, predicted = torch.max(outputs, 1)
            
            # 收集預測與標籤
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 轉換為 numpy 陣列
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # 計算指標
    avg_loss = total_loss / len(data_loader.dataset) if criterion else 0.0
    accuracy = calculate_accuracy(all_predictions, all_labels)
    
    return avg_loss, accuracy, all_predictions, all_labels


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    使用 seaborn 熱力圖繪製混淆矩陣
    注意: 圖表標籤使用英文以符合國際慣例
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        class_names: 類別名稱列表 (例如: ['0', '1', ..., '9'])
        save_path: 儲存圖表的路徑 (可選)
        figsize: 圖表大小
    """
    # 計算混淆矩陣
    cm = confusion_matrix(y_true, y_pred)
    
    # 建立圖表
    plt.figure(figsize=figsize)
    
    # 繪製熱力圖
    sns.heatmap(
        cm,
        annot=True,  # 在格子中顯示數字
        fmt='d',  # 整數格式
        cmap='Blues',  # 配色方案
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    # 設定標籤與標題 (使用英文)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    # 儲存
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
    印出詳細的分類報告，包含 precision, recall 與 F1-score
    
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
    
    print('\n指標說明:')
    print('- Precision (精確率): 預測為正例中，實際為正例的比例')
    print('- Recall (召回率): 實際為正例中，被正確預測的比例')
    print('- F1-Score: Precision 與 Recall 的調和平均數')
    print('- Support: 各類別的樣本數量')


def visualize_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    num_samples: int = 16,
    save_path: Optional[str] = None
) -> None:
    """
    視覺化模型在樣本影像上的預測結果
    圖表標題使用英文
    
    Args:
        model: 訓練好的模型
        data_loader: DataLoader
        device: 運算裝置
        class_names: 類別名稱列表
        num_samples: 要顯示的樣本數量
        save_path: 儲存圖表的路徑 (可選)
    """
    model.eval()
    
    # 取得一個批次的資料
    images, labels = next(iter(data_loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    # 進行預測
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)
    
    # 移至 CPU 以進行視覺化
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
        
        # 顯示影像 (反正規化)
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
        
        title = f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)'
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
    
    # 隱藏多餘的子圖
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Model Predictions', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'🖼️  預測結果已儲存至 {save_path}')
    
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
    視覺化模型錯誤分類的樣本
    
    Args:
        model: 訓練好的模型
        data_loader: DataLoader
        device: 運算裝置
        class_names: 類別名稱列表
        num_samples: 要顯示的錯誤樣本數量
        save_path: 儲存圖表的路徑 (可選)
    """
    model.eval()
    
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    misclassified_confs = []
    
    # 收集錯誤分類的樣本
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
            
            # 找出錯誤分類的樣本
            misclassified_mask = predictions != labels
            
            if misclassified_mask.sum() > 0:
                misclassified_images.append(images[misclassified_mask].cpu())
                misclassified_labels.append(labels[misclassified_mask].cpu())
                misclassified_preds.append(predictions[misclassified_mask].cpu())
                misclassified_confs.append(confidences[misclassified_mask].cpu())
            
            # 如果已經收集足夠的樣本就停止
            total_misclassified = sum(img.size(0) for img in misclassified_images)
            if total_misclassified >= num_samples:
                break
    
    if len(misclassified_images) == 0:
        print('🎉 沒有找到錯誤分類的樣本！模型表現完美！')
        return
    
    # 合併所有錯誤樣本
    misclassified_images = torch.cat(misclassified_images, dim=0)[:num_samples]
    misclassified_labels = torch.cat(misclassified_labels, dim=0)[:num_samples]
    misclassified_preds = torch.cat(misclassified_preds, dim=0)[:num_samples]
    misclassified_confs = torch.cat(misclassified_confs, dim=0)[:num_samples]
    
    actual_num = misclassified_images.size(0)
    
    # 繪製結果
    rows = int(np.sqrt(actual_num))
    cols = int(np.ceil(actual_num / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten() if actual_num > 1 else [axes]
    
    for idx in range(actual_num):
        ax = axes[idx]
        
        # 顯示影像
        img = misclassified_images[idx].squeeze()
        mean, std = 0.1307, 0.3081
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        # 設定標題（全部用紅色，因為都是錯誤）
        true_label = class_names[misclassified_labels[idx].item()]
        pred_label = class_names[misclassified_preds[idx].item()]
        confidence = misclassified_confs[idx].item() * 100
        
        title = f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)'
        ax.set_title(title, fontsize=10, color='red', fontweight='bold')
    
    # 隱藏多餘的子圖
    for idx in range(actual_num, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Misclassified Samples', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'🖼️  錯誤分類樣本已儲存至 {save_path}')
    
    plt.show()


if __name__ == '__main__':
    print('✅ 評估模組載入成功!')
    print('\n可用函式:')
    print('- calculate_accuracy(): 計算分類準確率')
    print('- evaluate_model(): 在資料集上評估模型')
    print('- plot_confusion_matrix(): 視覺化混淆矩陣')
    print('- print_classification_report(): 印出詳細指標')
    print('- visualize_predictions(): 視覺化模型預測')
    print('- visualize_misclassified(): 視覺化錯誤分類樣本')
    print('\n💡 這些函式已完整實作，可直接使用！')
    print('   學生的主要實作重點在 train.py 的訓練與驗證迴圈。')
