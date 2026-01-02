# Lab 2: CNN 特徵提取 (CIFAR-10)

## 📖 作業背景

本作業使用 **卷積神經網路 (Convolutional Neural Network, CNN)** 對 CIFAR-10 彩色影像資料集進行分類。CIFAR-10 包含 60,000 張 32x32 像素的 RGB 彩色影像，分為 10 個類別（飛機、汽車、鳥、貓、鹿、狗、青蛙、馬、船、卡車）。

相比 Lab 1 的 MNIST 灰階影像，CIFAR-10 具有更高的複雜度和更豐富的顏色資訊，因此需要使用 CNN 來有效提取空間特徵。

## 🎯 學習目標

透過本作業，你將學習到：

1. **卷積層 (Conv2d)**: 理解如何提取影像的局部特徵
2. **池化層 (Pooling)**: 學習降維與特徵摘要的作用
3. **資料增強 (Data Augmentation)**: 實作防止過擬合的技術
4. **多通道處理**: 理解 RGB 影像的 3 個顏色通道
5. **層次化特徵**: 認識 CNN 如何從低階到高階提取特徵
6. **Global Average Pooling**: 學習替代 Flatten 的先進技術

## 🛠 環境需求

### Python 版本
- Python 3.8+

### 相依套件
```bash
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
seaborn>=0.12.0
tqdm>=4.65.0
jupyter>=1.0.0
```

### 安裝指令
```bash
pip install -r requirements.txt  # 從根目錄執行
```

## 📁 檔案結構

```
Lab2-CNN/
├── data/                   # CIFAR-10 資料集 (自動下載)
├── dataset.py              # 資料集載入與增強
├── model.py                # CNN 模型架構
├── train.py                # 訓練器類別
├── eval.py                 # 評估指標與視覺化
├── utils.py                # 輔助函式
├── main.ipynb              # 主程式 Notebook
├── checkpoints/            # 模型檢查點 (自動建立)
├── results/                # 結果圖表 (自動建立)
└── README.md               # 本說明文件
```

## 🚀 快速開始

### 步驟 1: 啟動 Jupyter Notebook

```bash
cd Lab2-CNN
jupyter notebook main.ipynb
```

### 步驟 2: 完成學生實作

本 Lab 的訓練流程已完整提供，主要學習重點在於：

#### 1. 理解 CNN 架構 (`model.py`)
- 觀察 `ConvBlock` 的組成
- 理解 Tensor 形狀如何變化
- 學習 BatchNorm 和 Dropout 的作用

#### 2. 認識資料增強 (`dataset.py`)  
- 理解 RandomCrop 的效果
- 學習 RandomHorizontalFlip
- 觀察正規化的重要性

#### 3. 訓練迴圈 (`train.py`) - 與 Lab1 相同
- 5 個訓練步驟（如需複習可參考 Lab1）
- 驗證迴圈的實作

### 步驟 3: 執行訓練

依序執行 Notebook 中的 Cell：
- 系統會自動下載 CIFAR-10 資料集
- 建立 CNN 模型
- 訓練 50 個 epochs
- 產生訓練曲線與評估結果

### 步驟 4: 查看結果

訓練完成後，檢查 `./results` 目錄：
- `training_curves.png` - 訓練與驗證曲線
- `confusion_matrix.png` - 混淆矩陣
- `predictions.png` - 預測結果範例

## 📚 核心概念說明

### CNN 架構

```
輸入 RGB 影像 (3, 32, 32)
   ↓ Conv2d(3→64) + BatchNorm + ReLU + MaxPool
ConvBlock 1 (64, 16, 16)
   ↓ Conv2d(64→128) + BatchNorm + ReLU + MaxPool
ConvBlock 2 (128, 8, 8)
   ↓ Conv2d(128→256) + BatchNorm + ReLU + MaxPool
ConvBlock 3 (256, 4, 4)
   ↓ Conv2d(256→512) + BatchNorm + ReLU + MaxPool
ConvBlock 4 (512, 2, 2)
   ↓ Global Average Pooling
特徵向量 (512)
   ↓ Dropout + Linear
輸出 (10 個類別)
```

每個 ConvBlock 包含：
- **Conv2d**: 卷積層，提取空間特徵
- **BatchNorm2d**: 批次正規化，穩定訓練
- **ReLU**: 激活函數
- **MaxPool2d**: 最大池化，降低空間維度

### 為什麼使用 CNN 而非 MLP？

**MLP 的問題**:
- 展平影像會完全**損失空間結構**
- 相鄰像素的關係被破壞
- 參數量極大（32×32×3 = 3072 個輸入）

**CNN 的優勢**:
- 透過**卷積核**保留空間資訊
- **參數共享**大幅減少參數量
- **局部感受野**捕捉局部特徵
- 透過**池化**逐步抽象，提取高階特徵

### 資料增強 (Data Augmentation)

訓練時使用的增強技術：

```python
# 訓練集增強
transforms.RandomCrop(32, padding=4)      # 隨機裁切
transforms.RandomHorizontalFlip(p=0.5)    # 隨機水平翻轉
transforms.Normalize(mean, std)            # 正規化
```

**為什麼需要資料增強？**
- 增加訓練資料的多樣性
- 防止模型記憶訓練資料（過擬合）
- 提升模型的泛化能力

### Global Average Pooling vs Flatten

**Flatten** (Lab1 使用):
```
(512, 2, 2) → (2048)  # 直接展平
```

**Global Average Pooling** (Lab2 使用):
```
(512, 2, 2) → (512, 1, 1) → (512)  # 對每個通道取平均
```

**GAP 的優勢**:
- 參數更少（不需要大型 Linear 層）
- 對輸入尺寸更靈活
- 減少過擬合風險

### 超參數說明

- **base_channels**: 第一層的通道數（預設 64）
- **batch_size**: 每批次的樣本數量（預設 128）
- **learning_rate**: 學習率（預設 0.001）
- **num_epochs**: 訓練輪數（預設 50）
- **dropout_rate**: Dropout 機率（預設 0.5）

## 📝 學生作業

本作業需要你完成 **CNN 模型的實作**，並理解卷積神經網路的工作原理。

> 💡 **重要提示**: 每個檔案中都有詳細的逐步實作說明，請仔細閱讀程式碼中的註解！

---

### 📄 作業 1: 實作 ConvBlock (`model.py`)

**任務**: 建立標準的卷積區塊

**需要完成**:實作 `ConvBlock.__init__()` 方法

**實作提示**:
按順序建立以下層：
1. **Conv2d**: 卷積層提取空間特徵
   - `nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)`
2. **BatchNorm2d**: 批次正規化穩定訓練
   - `nn.BatchNorm2d(out_channels)`
3. **ReLU**: 激活函數引入非線性
   - `nn.ReLU(inplace=True)`
4. **MaxPool2d**: 最大池化降低維度（如果 pool=True）
   - `nn.MaxPool2d(kernel_size=2, stride=2)`

**組合方式**:
使用 `nn.Sequential` 將所有層組合為 `self.block`

**難度**: ⭐⭐  
**預計時間**: 20-30 分鐘

---

### 📄 作業 2: 實作 CNN 模型 (`model.py`)

**任務**: 組合多個 ConvBlock 建立完整的 CNN

**需要完成**:
實作 `CNN.__init__()` 方法中的三個部分

#### 2.1 建立特徵提取器 (`self.features`)
**實作提示**:
- 使用 `nn.Sequential` 組合 4 個 ConvBlock
- 通道數依序: 3 → 64 → 128 → 256 → 512
- 空間尺寸依序: 32×32 → 16×16 → 8×8 → 4×4 → 2×2

```python
self.features = nn.Sequential(
    ConvBlock(in_channels, base_channels, pool=True),
    ConvBlock(base_channels, base_channels * 2, pool=True),
    ConvBlock(...),  # 提示: base_channels * 2 → base_channels * 4
    ConvBlock(...)   # 提示: base_channels * 4 → base_channels * 8
)
```

#### 2.2 建立全域平均池化 (`self.global_avg_pool`)
**實作提示**:
- 使用 `nn.AdaptiveAvgPool2d((1, 1))`
- 將 (512, 2, 2) → (512, 1, 1)

#### 2.3 建立分類器 (`self.classifier`)
**實作提示**:
- 使用 `nn.Sequential` 組合 Dropout 和 Linear
- Dropout 防止過擬合
- Linear 輸出 10 個類別分數

```python
self.classifier = nn.Sequential(
    nn.Dropout(dropout_rate),
    nn.Linear(base_channels * 8, num_classes)
)
```

**難度**: ⭐⭐⭐  
**預計時間**: 40-50 分鐘

---

### 📄 作業 3: 理解資料增強 (`dataset.py`)

**學習任務**: 觀察資料增強的效果

#### 3.1 比較增強前後的影像
**實驗步驟**:
1. 載入訓練集 with 資料增強
2. 載入訓練集 without 資料增強
3. 視覺化對比

**觀察**: RandomCrop 和 RandomHorizontalFlip 如何改變影像

#### 3.2 理解增強的作用
**思考**:
- 為什麼測試集不需要資料增強？
- 資料增強如何防止過擬合？

**難度**: ⭐  
**預計時間**: 15-20 分鐘

---

### 📄 作業 4: 訓練與分析 (`train.py`, `eval.py`)

**任務**: 訓練 CNN 並分析結果

#### 4.1 訓練模型
**觀察項目**:
- Loss 下降曲線
- 訓練準確率 vs 驗證準確率
- 是否出現過擬合？

#### 4.2 分析混淆矩陣
**分析**:
- 哪些類別最容易混淆？
- Cat 和 Dog 容易混淆嗎？為什麼？
- Automobile 和 Truck 容易混淆嗎？

#### 4.3 觀察錯誤案例
**使用**: `visualize_predictions()` 函式
**思考**: 為什麼模型會犯這些錯誤？

**難度**: ⭐⭐  
**預計時間**: 30-40 分鐘

---

## 📋 實作順序建議

1. **實作 ConvBlock** - 建立基礎卷積區塊（30 分鐘）
2. **實作 CNN 模型** - 組合完整架構（50 分鐘）
3. **測試模型** - 執行 `python model.py` 確認正確（10 分鐘）
4. **執行訓練** - 在 `main.ipynb` 中訓練（30-60 分鐘）
5. **分析結果** - 觀察混淆矩陣與錯誤案例（20 分鐘）

## 💡 實作小技巧

- **理解層的順序** → Conv → BN → ReLU → Pool 是標準組合
- **追蹤形狀變化** → 在每層後加 `print(x.shape)` 觀察
- **測試 ConvBlock** → 先單獨測試 ConvBlock 再組合成 CNN
- **對比 MLP** → 思考為何 CNN 比 Lab1 的 MLP 更適合影像

## ⚠️ 常見錯誤提醒

1. **通道數順序錯誤** → 第二層的輸入通道要等於第一層的輸出通道
2. **忘記 pool 參數** → 所有 ConvBlock 都要設 `pool=True`
3. **分類器輸入維度錯誤** → 應該是 `base_channels * 8` (512)
4. **全域平均池化忘記設定** → 必須是 `nn.AdaptiveAvgPool2d((1, 1))`
5. **模型沒實作就執行** → 會出現 `NotImplementedError`

## 🎯 預期結果

完成訓練後，模型應達到：
- **訓練準確率**: ~85-90%
- **測試準確率**: ~75-80%
- **訓練時間**: ~30-60 分鐘（取決於硬體）

> **重要**: CIFAR-10 比 MNIST 困難得多，75-80% 已是良好表現！
> 
> 進階模型（ResNet, VGG）可達 90%+，但需要更複雜的架構

## 🤔 思考問題

1. 為什麼 CNN 比 MLP 更適合處理影像？試著從參數量和空間結構兩方面解釋
2. 資料增強如何幫助模型泛化？為什麼測試集不需要資料增強？
3. 觀察混淆矩陣，哪些類別最容易被混淆？為什麼？（提示：想想視覺相似性）
4. Global Average Pooling 相比 Flatten 有什麼優勢？在參數量上差多少？
5. 如何進一步提升準確率到 85%+？（提示：ResNet, 更深的網路, 更多 epochs, 學習率調整）
6. 如果輸入影像從 32×32 變成 64×64，模型需要改動哪些地方？

## 📖 參考資源

- [PyTorch CNN Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Understanding CNNs - CS231n](https://cs231n.github.io/convolutional-networks/)
- [Data Augmentation Explained](https://pytorch.org/vision/stable/transforms.html)
- [Global Average Pooling](https://paperswithcode.com/method/global-average-pooling)

## 💡 疑難排解

### CUDA Out of Memory
**解決方法**: 
- 減少 `batch_size` (128 → 64 → 32)
- 減少 `base_channels` (64 → 32)

### 訓練準確率很高但測試準確率很低
**原因**: 過擬合 (Overfitting)
**解決方法**:
- 增加 `dropout_rate` (0.5 → 0.6 → 0.7)
- 使用更強的資料增強
- 減少模型複雜度
- Early stopping

### 訓練太慢
**解決方法**: 
1. 確認是否使用 GPU/MPS
2. 減少 `num_epochs` (50 → 20) 先測試
3. 增加 `batch_size` (如果記憶體夠用)
4. 減少模型大小

### Loss 不下降或震盪
**解決方法**:
- 降低學習率 (0.001 → 0.0001)
- 檢查資料是否正確載入
- 確認模型沒有錯誤

## ✅ 作業提交檢查清單

- [ ] 實作 `ConvBlock.__init__()` (所有TODO區塊)
- [ ] 實作 `CNN.__init__()` (特徵提取器、全域平均池化、分類器)
- [ ] 成功訓練 CNN 模型
- [ ] 達到 >70% 測試準確率
- [ ] 理解 Conv2d 和 Pooling 的作用
- [ ] 理解資料增強的重要性
- [ ] 觀察並分析混淆矩陣
- [ ] 回答所有思考問題
- [ ] 比較 CNN 與 MLP 的差異

---

**祝學習順利！🎉**

**遇到問題？** 查看程式碼中的詳細註解，或參考 Lab1 的訓練流程。
