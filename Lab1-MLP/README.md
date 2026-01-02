# Lab 1: MLP 影像分類 (MNIST)

## 📖 作業背景

本作業使用 **多層感知機 (Multi-Layer Perceptron, MLP)** 對 MNIST 手寫數字資料集進行分類。MNIST 是電腦視覺領域的經典入門資料集，包含 60,000 張訓練影像與 10,000 張測試影像，每張影像為 28x28 像素的灰階手寫數字 (0-9)。

## 🎯 學習目標

透過本作業，你將學習到：

1. **自訂資料集**: 如何從 JSON 檔案載入影像與標籤
2. **Tensor 展平**: 理解如何將 2D 影像展平為 1D 向量
3. **全連接層**: 實作線性層 (Linear Layers)
4. **訓練迴圈**: 完整的訓練與驗證流程
5. **評估指標**: 使用準確率與混淆矩陣評估模型表現

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
Pillow>=9.0.0
tqdm>=4.65.0
jupyter>=1.0.0
```

### 安裝指令
```bash
pip install -r requirements.txt  # 從根目錄執行
```

## 📁 檔案結構

```
Lab1-MLP/
├── data/                   # 資料集目錄
│   ├── train.json          # 訓練集標註檔
│   ├── test.json           # 測試集標註檔
│   ├── train/              # 訓練影像
│   └── test/               # 測試影像
├── prepare_mnist.py        # 資料準備腳本
├── dataset.py              # 資料集載入器
├── model.py                # MLP 模型
├── train.py                # 訓練器類別
├── eval.py                 # 評估指標
├── utils.py                # 輔助函式
├── main.ipynb              # 主程式 Notebook
├── checkpoints/            # 模型檢查點 (自動建立)
├── results/                # 結果圖表 (自動建立)
└── README.md               # 本說明文件
```

## 🚀 快速開始

### 步驟 1: 準備資料集

首先，執行準備腳本下載並轉換 MNIST 為 JSON 格式：

```bash
cd Lab1-MLP
python prepare_mnist.py
```

這將建立：
- `data/train.json` - 訓練集標註 (60,000 筆)
- `data/test.json` - 測試集標註 (10,000 筆)
- `data/train/*.png` - 訓練影像
- `data/test/*.png` - 測試影像

**JSON 格式範例:**
```json
{
  "num_samples": 60000,
  "num_classes": 10,
  "samples": [
    {
      "id": "00000",
      "image_path": "train/00000.png",
      "label": 5
    },
    ...
  ]
}
```

### 步驟 2: 完成學生實作

你需要實作以下檔案：

#### 1. `dataset.py` (自訂資料載入器)
- 實作 `__init__`: 載入 JSON 檔案
- 實作 `__len__`: 回傳資料集長度
- 實作 `__getitem__`: 載入並轉換單張影像
- 實作 `create_dataloaders()`: 建立 DataLoader

#### 2. `model.py` (MLP 模型)
- 實作 `__init__`: 建立網路層 (784→512→256→128→10)
- 實作 `forward`: 定義前向傳播，包含展平與逐層計算

#### 3. `train.py` (訓練與驗證)
- 實作 `train_one_epoch()`: 完成訓練迴圈的 5 個步驟
- 實作 `validate()`: 完成驗證迴圈

### 步驟 3: 執行訓練

```bash
jupyter notebook main.ipynb
```

依序執行 Cell 以：
- 載入資料集
- 建立 MLP 模型
- 訓練 20 個 epochs
- 評估與視覺化結果

### 步驟 4: 查看結果

訓練完成後，檢查 `./results` 目錄：
- `training_curves.png` - 訓練與驗證曲線
- `confusion_matrix.png` - 混淆矩陣
- `predictions.png` - 預測結果範例
- `metrics.json` - 訓練指標

## 📚 核心概念說明

### MLP 架構
```
輸入 (28x28)
   ↓ 展平
輸入層 (784)
   ↓ Linear → BatchNorm → ReLU → Dropout
隱藏層 1 (512)
   ↓ Linear → BatchNorm → ReLU → Dropout
隱藏層 2 (256)
   ↓ Linear → BatchNorm → ReLU → Dropout
隱藏層 3 (128)
   ↓ Linear
輸出層 (10)
```

每個隱藏層包含：
- **Linear**: 全連接層轉換
- **BatchNorm**: 批次正規化，穩定訓練
- **ReLU**: 激活函數 (ReLU(x) = max(0, x))
- **Dropout**: 隨機丟棄神經元，防止過擬合

### 為什麼需要展平？
MLP 只能處理 1D 向量，因此需要將 28x28 影像展平：
```
(batch_size, 1, 28, 28) → (batch_size, 784)
```

### 超參數說明
- **batch_size**: 每次訓練的樣本數量
- **learning_rate**: 權重更新的步長
- **dropout_rate**: 丟棄神經元的機率
- **num_epochs**: 訓練迭代次數

### 訓練迴圈 (Training Loop) - 核心概念
深度學習訓練的 5 個核心步驟：

1. **清空梯度** (`optimizer.zero_grad()`)
   - PyTorch 預設會累積梯度，必須手動清空

2. **前向傳播** (`outputs = model(images)`)
   - 輸入傳遞通過網路，得到預測

3. **計算損失** (`loss = criterion(outputs, labels)`)
   - 比較預測與真實標籤的差距

4. **反向傳播** (`loss.backward()`)
   - 計算損失對各參數的梯度

5. **更新權重** (`optimizer.step()`)
   - 根據梯度調整參數，使模型改進

**驗證與訓練的差異:**
- 驗證使用 `torch.no_grad()` 停用梯度計算
- 不執行步驟 1, 4, 5（只前向傳播和計算損失）

## 📝 學生作業

本作業需要你完成 **3 個 Python 檔案** 的實作，每個檔案都包含詳細的 TODO 註解指導。

> 💡 **重要提示**: 每個檔案中都有詳細的逐步實作說明，請仔細閱讀程式碼中的註解！

---

### 📄 作業 1: 實作資料載入器 (`dataset.py`)

**需要完成的函式:**

#### 1.1 `MNISTDataset.__init__()` 
**任務**: 從 JSON 檔案載入樣本列表

**實作提示**:
- 使用 `json.load()` 讀取 JSON 檔案
- 提取 `samples` 列表並儲存到 `self.samples`
- JSON 格式為: `{"samples": [{"id": "...", "image_path": "...", "label": ...}]}`

#### 1.2 `MNISTDataset.__len__()`
**任務**: 回傳資料集大小

**實作提示**:
- 回傳 `self.samples` 的長度
- 這讓 DataLoader 知道有多少資料

#### 1.3 `MNISTDataset.__getitem__(idx)`
**任務**: 載入單一影像與標籤

**實作提示**:
- 從 `self.samples[idx]` 取得影像路徑與標籤
- 使用 `Image.open()` 載入 PNG 影像
- 套用 `self.transform` 將影像轉為 Tensor
- 回傳 `(影像, 標籤)` tuple

#### 1.4 `create_dataloaders()`
**任務**: 建立訓練與測試的 DataLoader

**實作提示**:
- 建立兩個 `MNISTDataset` 實例 (train/test)
- 用 `DataLoader` 包裝，設定 `batch_size` 和 `shuffle`
- 訓練集要 `shuffle=True`，測試集 `shuffle=False`

**難度**: ⭐⭐  
**預計時間**: 30-40 分鐘

---

### 📄 作業 2: 實作 MLP 模型 (`model.py`)

**需要完成的函式:**

#### 2.1 `MLP.__init__()`
**任務**: 建立 MLP 的所有網路層

**實作提示**:
- 建立 4 組層: 3 個隱藏層 + 1 個輸出層
- 每個隱藏層包含: `Linear → BatchNorm → ReLU → Dropout`
- 架構: 784 → 512 → 256 → 128 → 10
- 將每層分別指派給 `self.fc1`, `self.bn1`, `self.relu1`, ... 等

#### 2.2 `MLP.forward(x)`
**任務**: 定義前向傳播流程

**實作提示**:
- **步驟 1**: 將輸入從 `(batch, 1, 28, 28)` 展平為 `(batch, 784)`
- **步驟 2-4**: 依序通過 3 個隱藏層，每層都要套用 fc → bn → relu → dropout
- **步驟 5**: 通過輸出層得到 10 個類別的分數
- **注意**: 不要加 softmax (CrossEntropyLoss 會處理)

**難度**: ⭐⭐  
**預計時間**: 40-50 分鐘

---

### 📄 作業 3: 實作訓練與驗證迴圈 (`train.py`)

**需要完成的函式:**

#### 3.1 `Trainer.train_one_epoch()` - 訓練迴圈
**任務**: 實作單一 epoch 的訓練流程

**實作提示**:
在 for 迴圈中完成 5 個訓練步驟:
1. **清空梯度**: `self.optimizer.zero_grad()`
2. **前向傳播**: `outputs = self.model(images)`
3. **計算損失**: `loss = self.criterion(outputs, labels)`
4. **反向傳播**: `loss.backward()`
5. **更新權重**: `self.optimizer.step()`

**為什麼這 5 步很重要?**
- 步驟 1: 防止梯度累積 (PyTorch 預設會累積)
- 步驟 2-3: 計算模型當前的預測誤差
- 步驟 4-5: 根據誤差調整模型參數，使預測更準確

#### 3.2 `Trainer.validate()` - 驗證迴圈
**任務**: 實作驗證流程 (與訓練類似但有關鍵差異)

**實作提示**:
與訓練迴圈類似，但有以下差異：
- ✅ 需要 `with torch.no_grad():` 包住整個迴圈
- ❌ 不需要 `optimizer.zero_grad()`
- ❌ 不需要 `loss.backward()`  
- ❌ 不需要 `optimizer.step()`

**為什麼驗證不同?**
- 驗證只評估模型表現，不更新參數
- `torch.no_grad()` 停用梯度計算，節省記憶體

**難度**: ⭐⭐⭐  
**預計時間**: 40-50 分鐘

---

## 📋 實作順序建議

建議按照以下順序完成作業：

1. **先跑 `prepare_mnist.py`** 準備資料集 (已完成，只需執行)
2. **實作 `dataset.py`** - 從載入資料開始
3. **實作 `model.py`** - 建立模型架構  
4. **實作 `train.py`** - 完成訓練與驗證迴圈
5. **執行 `main.ipynb`** - 整合訓練與測試

## 💡 實作小技巧

- **不知道怎麼開始?** → 每個檔案都有 `if __name__ == '__main__':` 區塊可以單獨測試
- **看不懂要做什麼?** → 程式碼中的 TODO 區塊有 **非常詳細** 的逐步說明
- **遇到錯誤?** → 檢查：
  - Tensor 形狀是否正確
  - 是否正確移動資料到 device
  - 函式回傳值是否符合預期型別
- **想確認正確性?** → 執行各檔案的測試區塊，會顯示預期的輸出
- **訓練迴圈寫對了嗎?** → 移除 `raise NotImplementedError` 後，應該能看到 loss 下降

## ⚠️ 常見錯誤提醒

1. **忘記清空梯度** → 每次迭代前一定要 `optimizer.zero_grad()`
2. **驗證時沒用 `no_grad`** → 會浪費記憶體且可能導致錯誤
3. **展平時忘記保留 batch 維度** → 使用 `x.view(batch_size, -1)` 而非 `x.view(-1)`
4. **忘記將資料移到 device** → 訓練/評估時都要 `.to(device)`
5. **在 forward 中加了 softmax** → 使用 CrossEntropyLoss 時不需要！

## 🎯 預期結果

完成實作並訓練後，模型應達到：
- **訓練準確率**: ~99%
- **測試準確率**: ~98%
- **訓練時間**: ~5-10 分鐘 (取決於硬體)

## 🤔 思考問題

1. 為什麼 MLP 需要展平影像？這會損失哪些資訊？
2. BatchNorm 和 Dropout 的作用分別是什麼？
3. 觀察混淆矩陣，哪些數字最容易被混淆？為什麼？
4. 如何進一步提升模型準確率？(提示：調整超參數、增加層數等)
5. MLP 的局限性是什麼？為何需要 CNN？

## 📖 參考資源

- [PyTorch 官方文件](https://pytorch.org/docs/stable/index.html)
- [MNIST 資料集](http://yann.lecun.com/exdb/mnist/)
- [Understanding MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron)

## 💡 疑難排解

### FileNotFoundError: train.json not found
**解決方法**: 先執行 `python prepare_mnist.py` 準備資料集

### CUDA Out of Memory
**解決方法**: 減少 `batch_size` (例如：從 64 降到 32)

### 訓練太慢
**解決方法**: 
1. 確認是否使用 GPU/MPS
2. 若在 Windows 上，減少 `num_workers`
3. 減少 `num_epochs` 進行測試

### NotImplementedError
**解決方法**: 完成程式碼檔案中的 TODO 區塊

## ✅ 作業提交檢查清單

- [ ] 實作 `dataset.py` (所有 TODO 區塊)
- [ ] 實作 `model.py` (所有 TODO 區塊)
- [ ] 實作 `train.py` (訓練與驗證迴圈) ⭐
- [ ] 成功訓練模型
- [ ] 達到 >95% 測試準確率
- [ ] 產生所有結果圖表
- [ ] 回答思考問題

---

**祝學習順利！🎉**

**遇到問題？** 查看各個 Python 檔案中的詳細註解，裡面有逐步指導。
