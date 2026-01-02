# Lab 4: 語義分割 (Semantic Segmentation) - Oxford-IIIT Pet

## 📖 作業背景

本作業介紹 **語義分割 (Semantic Segmentation)**，這是比物件偵測更精細的視覺任務。分割不僅要識別物體，還要**對每個像素**進行分類，產生與輸入影像相同尺寸的分割遮罩 (Segmentation Mask)。

我們使用 **U-Net** 架構處理 Oxford-IIIT Pet 資料集，該資料集包含貓狗的影像與對應的三分割遮罩（前景、背景、輪廓）。

> ⚠️ **重要說明**: 本 Lab 著重於 **U-Net 架構理解**
> 
> 完整的分割訓練需要大量標註資料與訓練時間。本 Lab 提供完整的 U-Net 實作與評估指標，幫助你理解分割任務的核心概念。實際應用建議使用 `torch vision.datasets.OxfordIIITPet` 載入資料。

## 🎯 學習目標

透過本作業，你將學習到：

1. **語義分割概念**: 理解像素級別的影像理解
2. **U-Net 架構**: 學習 Encoder-Decoder 的對稱設計
3. **Skip Connections**: 理解如何融合不同層級的特徵
4. **Dice Score**: 學習分割任務的專用評估指標
5. **mIoU (mean IoU)**: 理解多類別 IoU 的計算
6. **上採樣技術**: 認識如何恢復空間解析度

## 🛠 環境需求

### Python 版本
- Python 3.8+

### 相依套件
```bash
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
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
Lab4-Segment/
├── data/                   # Oxford Pet 資料集目錄
├── dataset.py              # 資料集載入（建議用 torchvision）
├── model.py                # U-Net 模型實作
├── train.py                # 訓練器類別
├── eval.py                 # Dice, mIoU 評估指標
├── utils.py                # 輔助函式
├── main.ipynb              # 主程式 Notebook
└── README.md               # 本說明文件
```

## 🚀 快速開始

### 步驟 1: 理解任務差異

**影像分類** (Lab 1 & 2):
- 輸入: (3, H, W)
- 輸出: 單一類別標籤

**物件偵測** (Lab 3):
- 輸入: (3, H, W)
- 輸出: 多個 Bounding Boxes + 類別

**語義分割** (Lab 4):
- 輸入: (3, H, W)
- 輸出: (H, W) 分割遮罩，每個像素都有類別標籤

### 步驟 2: 理解 U-Net 架構

本 Lab 的核心是理解 U-Net 的設計：
- **Encoder**: 逐步降低解析度，提取特徵
- **Decoder**: 逐步恢復解析度，生成分割遮罩
- **Skip Connections**: 連接 Encoder 與 Decoder，保留細節

### 步驟 3: 學習評估指標

分割任務使用專門的指標：
- **Dice Score**: 衡量預測與真實遮罩的重疊程度
- **Pixel Accuracy**: 像素級別的準確率
- **mIoU**: 多類別的平均 IoU

### 步驟 4: 執行學習

```bash
cd Lab4-Segment
jupyter notebook main.ipynb
```

## 📚 核心概念說明

### 語義分割 vs 物件偵測

```
原始影像          物件偵測             語義分割
┌─────────┐      ┌─────────┐        ┌─────────┐
│  🐱    │      │ ┌─┐     │        │████     │
│     🐶 │  →   │ │🐱│ 🐶  │   →    │████ ███ │
│         │      │ └─┘ └─┘ │        │     ███ │
└─────────┘      └─────────┘        └─────────┘
               (矩形框)            (像素級遮罩)
```

### U-Net 架構

```
輸入 (3, 256, 256)
   ↓
Encoder (下採樣)
┌─────────────────┐
│ Conv+Pool       │ ─┐
│ (64, 128, 128)  │  │
├─────────────────┤  │
│ Conv+Pool       │  │ Skip
│ (128, 64, 64)   │  │ Connections
├─────────────────┤  │
│ Conv+Pool       │  │
│ (256, 32, 32)   │  │
├─────────────────┤  │
│ Conv+Pool       │  │
│ (512, 16, 16)   │ ─┘
└─────────────────┘
   ↓
Bottleneck (1024, 8, 8)
   ↓
Decoder (上採樣)
┌─────────────────┐
│ UpConv + Concat │ ←┐
│ (512, 16, 16)   │  │
├─────────────────┤  │
│ UpConv + Concat │  │ 從 Encoder
│ (256, 32, 32)   │  │ 接收特徵
├─────────────────┤  │
│ UpConv + Concat │  │
│ (128, 64, 64)   │  │
├─────────────────┤  │
│ UpConv + Concat │  │
│ (64, 128, 128)  │ ←┘
└─────────────────┘
   ↓
輸出層 (num_classes, 256, 256)
```

### Encoder-Decoder 架構

**Encoder (編碼器)**:
- 透過卷積 + 池化逐步**降低空間解析度**
- **增加通道數**，提取更抽象的特徵
- 256×256 → 128×128 → 64×64 → 32×32 → 16×16

**Decoder (解碼器)**:
- 透過上採樣 + 卷積逐步**恢復空間解析度**
- **減少通道數**，重建分割遮罩
- 16×16 → 32×32 → 64×64 → 128×128 → 256×256

**為何需要 Skip Connections?**
- Encoder 會損失空間細節
- Skip Connections 將 Encoder 的細節特徵傳給 Decoder
- 有助於恢復物體邊界等精細資訊

### Skip Connections 機制

```python
# Encoder
enc1 = encoder_block1(x)     # (64, 128, 128)
enc2 = encoder_block2(enc1)  # (128, 64, 64)

# Decoder
dec2 = decoder_block2(bottleneck)  # (128, 64, 64)
dec2 = torch.cat([dec2, enc2], dim=1)  # 拼接！
dec1 = decoder_block1(dec2)  # (64, 128, 128)
dec1 = torch.cat([dec1, enc1], dim=1)  # 拼接！
```

**拼接的效果**:
- Decoder 特徵: 抽象的語義資訊
- Encoder 特徵: 精細的空間資訊
- 兩者結合 = 既準確又精細的分割

### Dice Score

```
預測遮罩: ████      真實遮罩: ████
          ████                 ████
          
交集: ████
聯集: ████████

Dice = 2 × |交集| / (|預測| + |真實|)
     = 2 × 4 / (6 + 6)
     = 8 / 12
     ≈ 0.67
```

**Dice Score 的特點**:
- 範圍: 0 (完全不重疊) ~ 1 (完全重疊)
- 對小物體更敏感（比 IoU 更適合分割）
- 常用作損失函數 (Dice Loss)

### mIoU (mean Intersection over Union)

對每個類別計算 IoU，然後取平均：

```python
class_0_iou = IoU(pred_class_0, true_class_0)  # 背景
class_1_iou = IoU(pred_class_1, true_class_1)  # 前景
class_2_iou = IoU(pred_class_2, true_class_2)  # 輪廓

mIoU = (class_0_iou + class_1_iou + class_2_iou) / 3
```

## 📝 學生作業

本作業著重於**理解 U-Net 架構**。

> 💡 **學習重點**: 理解 Encoder-Decoder 設計，認識 Skip Connections 的作用

---

### 📄 作業 1: 理解 U-Net 架構 (`model.py`)

**學習任務**:

#### 1.1 追蹤 Encoder 路徑
**重點**: 理解下採樣過程

**觀察項目**:
```python
# 在 model.py 中找到
輸入: (3, 256, 256)
  ↓ enc1
(64, 128, 128)    # 通道 3→64, 空間 256→128
  ↓ enc2  
(128, 64, 64)     # 通道 64→128, 空間 128→64
  ↓ enc3
(256, 32, 32)     # 通道 128→256, 空間 64→32
  ↓ enc4
(512, 16, 16)     # 通道 256→512, 空間 32→16
```

**思考**:
- 為何通道數不斷增加？
- 為何空間尺寸不斷減小？
-這與 CNN 的特徵提取有何關聯？

#### 1.2 理解 Skip Connections
**重點**: 觀察特徵拼接

**找到程式碼**:
```python
# 在 forward() 中
dec4 = self.upconv4(bottleneck)    # (512, 16, 16)
dec4 = torch.cat([dec4, enc4], dim=1)  # (1024, 16, 16)
#                        ↑
#                   來自 Encoder！
```

**思考**:
- 為何拼接後通道數變兩倍？
- 拼接發生在哪個維度？(dim=1 是通道維度)
- Skip Connection 傳遞了什麼資訊？

#### 1.3 追蹤 Decoder 路徑
**重點**: 理解上採樣過程

**觀察項目**:
- 使用 `ConvTranspose2d` 進行上採樣
- 空間尺寸逐步恢復: 16→32→64→128→256
- 通道數逐步減少: 512→256→128→64

**難度**: ⭐⭐⭐  
**預計時間**: 40-50 分鐘

---

### 📄 作業 2: 理解評估指標 (`eval.py`)

**學習任務**:

#### 2.1 Dice Score 計算
**重點**: 理解重疊度量

**觀察實作**:
```python
def calculate_dice(pred, target):
    intersection = (pred * target).sum()
    dice = 2 * intersection / (pred.sum() + target.sum())
    return dice
```

**思考**:
- 為何分子要乘以 2？
- Dice Score 與 IoU 有何不同？
- 為何分割任務偏好 Dice？

#### 2.2 mIoU 計算
**重點**: 多類別評估

**理解流程**:
1. 對每個類別分別計算 IoU
2. 忽略不存在的類別
3. 取所有類別的平均

**難度**: ⭐⭐  
**預計時間**: 20-30 分鐘

---

### 📄 作業 3: 視覺化分割結果 (`eval.py`)

**學習任務**:

#### 3.1 觀察分割遮罩
**重點**: 理解模型輸出

**使用函式**:
```python
visualize_segmentation(
    model, test_loader, device, 
    num_samples=4
)
```

**觀察**:
- 模型如何區分前景與背景？
- 邊界是否清晰？
- 哪些區域容易出錯？

#### 3.2 分析錯誤案例
**重點**: 理解模型局限

**思考**:
- 哪些類型的影像分割效果較差？
- 小物體的分割準確嗎？
- 物體邊界是否精確？

**難度**: ⭐  
**預計時間**: 15-20 分鐘

---

## 📋 學習順序建議

1. **閱讀 U-Net 架構** - 理解 Encoder-Decoder（30 分鐘）
2. **追蹤 Tensor 形狀** - 在程式碼中列印每層的形狀（20 分鐘）
3. **理解 Skip Connections** - 觀察 `torch.cat` 的作用（20 分鐘）
4. **學習評估指標** - Dice Score 和 mIoU（30 分鐘）
5. **視覺化結果** - 觀察分割遮罩（20 分鐘）
6. **（選做）嘗試訓練** - 在小資料集上訓練（1-2 小時）

## 💡 實作小技巧

- **追蹤形狀變化** → 在每個 block 後加 `print(x.shape)`
- **視覺化 Skip Connections** → 畫圖理解 Encoder 和 Decoder 的對應關係
- **理解上採樣** → 對比 `nn.ConvTranspose2d` 和 `nn.Upsample`
- **觀察顏色編碼** → 不同類別用不同顏色顯示分割遮罩

## ⚠️ 常見問題提醒

1. **形狀不匹配錯誤** → Skip Connection 的拼接要求 Encoder 和 Decoder 空間尺寸相同
2. **記憶體需求高** → 分割需要處理完整解析度的影像，記憶體消耗大
3. **訓練時間長** → 比分類任務慢很多，需要更多 epochs
4. **本 Lab 著重理解** → 不要求完整訓練，重點是架構理解
5. **資料標註困難** → 分割需要像素級標註，成本遠高於分類

## 🎯 預期成果

完成本作業後，你應該能夠：

- ✅ 理解 U-Net 的 Encoder-Decoder 架構
- ✅ 解釋 Skip Connections 的作用
- ✅ 計算 Dice Score 和 mIoU
- ✅ 理解分割任務與分類/偵測的差異
- ✅ 知道上採樣的實現方式
- ✅ 認識分割任務的應用場景

## 🤔 思考問題

1. 為什麼 U-Net 需要對稱的 Encoder-Decoder 結構？
2. Skip Connections 傳遞的是什麼資訊？為何對分割任務很重要？
3. Dice Loss 相比 CrossEntropy Loss 有什麼優勢？何時使用？
4. 上採樣有哪些方法？ConvTranspose2d 和 Upsample+Conv 的差異？
5. 如何處理類別不平衡問題？（例如：背景佔 90%）
6. 語義分割有哪些實際應用？（醫學影像、自動駕駛等）

## 📖 參考資源

### 核心論文
- [U-Net](https://arxiv.org/abs/1505.04597) - 原始論文（醫學影像分割）
- [FCN](https://arxiv.org/abs/1411.4038) - 全卷積網路
- [DeepLab](https://arxiv.org/abs/1606.00915) - Atrous Convolution
- [PSPNet](https://arxiv.org/abs/1612.01105) - Pyramid Pooling

### 教學資源
- [PyTorch Segmentation Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [Understanding U-Net](https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47)
- [Dice Loss Explained](https://www.jeremyjordan.me/semantic-segmentation/)

## 💡 進階學習方向

完成本 Lab 後，建議的進階方向：

1. **嘗試其他架構**
   - DeepLabV3+ (Atrous Convolution)
   - PSPNet (Pyramid Pooling)
   - SegFormer (Transformer-based)

2. **改進 U-Net**
   - Attention U-Net (加入注意力機制)
   - Residual U-Net (使用殘差連接)
   - 3D U-Net (處理體積資料)

3. **實際應用**
   - 醫學影像分割（腫瘤、器官）
   - 自動駕駛（道路、行人、車輛）
   - 衛星影像分析（建築、植被、水體）

4. **使用預訓練模型**
```python
import segmentation_models_pytorch as smp
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3
)
```

## ✅ 作業提交檢查清單

- [ ] 理解 U-Net 的 Encoder-Decoder 架構
- [ ] 能夠解釋 Skip Connections 的作用
- [ ] 理解 Dice Score 和 mIoU 的計算
- [ ] 追蹤過完整的 Tensor 形狀變化
- [ ] 觀察並分析分割結果
- [ ] 知道上採樣的實現方式
- [ ] 回答所有思考問題
- [ ] （選做）在小資料集上嘗試訓練

---

**祝學習順利！🎉**

**重要提醒**: U-Net 是分割任務的經典架構，理解其設計思想比完整訓練更重要。實際應用時可使用預訓練模型或專門的分割庫（如 `segmentation_models_pytorch`）。
