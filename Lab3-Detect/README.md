# Lab 3: 物件偵測 (Object Detection) - PASCAL VOC

## 📖 作業背景

本作業介紹 **物件偵測 (Object Detection)** 的核心概念與評估方法。物件偵測不僅要識別影像中的物體類別，還需要**定位物體的位置**（使用 Bounding Box）。

> ⚠️ **重要說明**: 本 Lab 著重於**概念理解**而非完整實作
> 
> 物件偵測的完整實作（如 Faster R-CNN, YOLO）非常複雜，不適合初學者從零開始。本 Lab 提供簡化版模型與核心演算法（IoU, NMS），幫助你理解物件偵測的基本原理。

實際應用建議使用 `torchvision` 的預訓練模型：
```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn
model = fasterrcnn_resnet50_fpn(pretrained=True)
```

## 🎯 學習目標

透過本作業，你將學習到：

1. **Bounding Box**: 理解如何用座標表示物體位置
2. **IoU (Intersection over Union)**: 學習計算兩個框的重疊程度
3. **NMS (Non-Maximum Suppression)**: 實作移除重複偵測的演算法
4. **mAP (mean Average Precision)**: 理解物件偵測的評估指標
5. **偵測模型架構**: 認識 Backbone, Classification Head, BBox Head 的設計
6. **與分類任務的差異**: 理解為何物件偵測更困難

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
Lab3-Detect/
├── data/                   # PASCAL VOC 資料集目錄
├── dataset.py              # 資料集載入（建議用 torchvision）
├── model.py                # 簡化版偵測模型（教學用）
├── train.py                # 訓練器類別
├── eval.py                 # IoU, NMS, mAP 實作
├── utils.py                # 輔助函式
├── main.ipynb              # 主程式 Notebook
└── README.md               # 本說明文件
```

## � 快速開始

### 步驟 1: 理解核心概念

物件偵測與分類的差異：

**分類任務** (Lab 1 & 2):
- 輸入: 影像
- 輸出: 類別標籤（例如: "貓"）

**物件偵測任務** (Lab 3):
- 輸入: 影像
- 輸出: 
  - 多個物體的**類別標籤**
  - 每個物體的**Bounding Box 座標** (x1, y1, x2, y2)
  - 每個偵測的**信心分數**

### 步驟 2: 學習核心演算法

本 Lab 重點在於理解與實作：

#### 1. **IoU (Intersection over Union)** - `eval.py`
計算兩個 Bounding Box 的重疊程度

```python
def calculate_iou(box1, box2):
    # 計算交集區域
    # 計算聯集區域  
    # IoU = 交集 / 聯集
    return iou
```

#### 2. **NMS (Non-Maximum Suppression)** - `eval.py`
移除重複的偵測框

```python
def non_maximum_suppression(boxes, scores, threshold):
    # 1. 按分數排序
    # 2. 選取最高分的框
    # 3. 移除與其高度重疊的框
    # 4. 重複直到所有框處理完
    return kept_boxes
```

### 步驟 3: 執行學習

```bash
cd Lab3-Detect
jupyter notebook main.ipynb
```

## 📚 核心概念說明

### Bounding Box 表示法

```
(x1, y1) ────────┐
   │             │
   │   物體      │  
   │             │
   └──────── (x2, y2)

座標格式: [x1, y1, x2, y2]
- (x1, y1): 左上角座標
- (x2, y2): 右下角座標
```

### IoU (Intersection over Union)

```
框A: ┌───────┐
     │   ┌───┼───┐  框B
     │   │█████   │
     └───┼───┘   │
         └───────┘

IoU = 交集面積(█) / 聯集面積(A∪B)
```

**IoU 的意義**:
- IoU = 1.0: 完全重疊
- IoU = 0.5: 中等重疊（常用閾值）
- IoU = 0.0: 完全不重疊

**應用**:
- 評估預測框與真實框的匹配程度
- NMS 中判斷是否為重複偵測

### NMS (Non-Maximum Suppression)

**問題**: 模型可能對同一物體產生多個偵測框

```
Before NMS:          After NMS:
  ┌──┐                 ┌──┐
  │┌─┼─┐               │  │
  ││ │ │      →        │  │
  └┼─┘ │               └──┘
   └───┘             (Only keep highest score)
```

**演算法步驟**:
1. 將所有偵測框按信心分數排序
2. 選取分數最高的框
3. 移除所有與該框 IoU > 閾值的框
4. 重複 2-3 直到沒有框剩餘

### mAP (mean Average Precision)

物件偵測的標準評估指標，綜合考慮：
- Precision: 預測為正例中真正為正例的比例
- Recall: 實際為正例中被正確預測的比例

**計算概念** （簡化說明）:
1. 對每個類別計算 AP (Average Precision)
2. AP 是 Precision-Recall 曲線下的面積
3. mAP 是所有類別 AP 的平均值

> 完整的 mAP 計算較複雜，建議參考外部資源理解細節

### 簡化版偵測模型架構

```
輸入影像
   ↓
Backbone (ResNet-like)
   - 提取特徵圖
   ↓
┌──────────┬──────────┐
│          │          │
Classification  BBox Regression
Head            Head
│          │          │
類別分數    框座標調整
└──────────┴──────────┘
   ↓
後處理 (NMS)
   ↓
最終偵測結果
```

## 📝 學生作業

本作業著重於**理解演算法**而非完整系統實作。

> 💡 **學習重點**: 理解 IoU 和 NMS 的運作原理，認識物件偵測的評估方法

---

### 📄 作業 1: 實作 IoU 計算 (`eval.py`)

**任務**: 完成 `calculate_iou()` 函式

**實作步驟**:

1. **計算交集區域**
```python
# 找出重疊區域的座標
x1 = max(box1[0], box2[0])  # 左邊界
y1 = max(box1[1], box2[1])  # 上邊界
x2 = min(box1[2], box2[2])  # 右邊界
y2 = min(box1[3], box2[3])  # 下邊界
```

2. **計算交集面積**
```python
# 注意: 如果無交集，寬度/高度會是負數
width = max(0, x2 - x1)
height = max(0, y2 - y1)
intersection = width * height
```

3. **計算聯集面積**
```python
area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
union = area1 + area2 - intersection
```

4. **計算 IoU**
```python
iou = intersection / union if union > 0 else 0
return iou
```

**測試案例**:
```python
box1 = [0, 0, 10, 10]  # 面積 100
box2 = [5, 5, 15, 15]  # 面積 100
# 交集: [5, 5, 10, 10] 面積 25
# 聯集: 100 + 100 - 25 = 175
# IoU = 25 / 175 ≈ 0.143
```

**難度**: ⭐⭐  
**預計時間**: 20-30 分鐘

---

### 📄 作業 2: 實作 NMS 演算法 (`eval.py`)

**任務**: 完成 `non_maximum_suppression()` 函式

**實作步驟**:

1. **按分數排序**
```python
# 將框按信心分數降序排列
indices = np.argsort(scores)[::-1]
boxes = boxes[indices]
scores = scores[indices]
```

2. **建立保留列表**
```python
kept_boxes = []
kept_scores = []
```

3. **迭代處理**
```python
while len(boxes) > 0:
    # 選取分數最高的框
    best_box = boxes[0]
    kept_boxes.append(best_box)
    
    # 計算與其他框的 IoU
    ious = [calculate_iou(best_box, box) for box in boxes[1:]]
    
    # 移除 IoU > threshold 的框
    boxes = boxes[1:][ious < threshold]
```

4. **回傳結果**
```python
return np.array(kept_boxes)
```

**難度**: ⭐⭐⭐  
**預計時間**: 30-40 分鐘

---

### 📄 作業 3: 理解 mAP 概念 (`eval.py`)

**任務**: 閱讀並理解 mAP 的計算概念

**學習重點**:

1. **Precision 與 Recall**
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)

2. **什麼是 True Positive?**
   - 預測框與真實框的 IoU > 0.5
   - 且類別預測正確

3. **為何需要 mAP?**
   - 單一準確率無法完整評估物件偵測
   - 需要同時考慮定位準確度（IoU）和分類準確度

**參考資源**:
- [GitHub: mAP Implementation](https://github.com/Cartucho/mAP)
- [Understanding mAP](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)

**難度**: ⭐⭐⭐ (概念性)  
**預計時間**: 30 分鐘（閱讀）

---

## 📋 學習順序建議

1. **理解 Bounding Box** - 座標表示法（5 分鐘）
2. **實作 IoU** - 交集與聯集計算（30 分鐘）
3. **測試 IoU** - 確認實作正確（10 分鐘）
4. **實作 NMS** - 迭代移除重複框（40 分鐘）
5. **閱讀 mAP** - 理解評估指標（30 分鐘）
6. **觀察簡化模型** - 理解架構設計（20 分鐘）

## 💡 實作小技巧

- **畫圖理解 IoU** → 在紙上畫出兩個框，手動計算交集與聯集
- **列印中間結果** → 在 NMS 中印出每一步保留/移除的框
- **視覺化驗證** → 用 matplotlib 畫出 NMS 前後的偵測框
- **參考完整實作** → 看 `torchvision.ops.nms` 的官方實作

## ⚠️ 常見問題提醒

1. **IoU 計算錯誤** → 注意無交集時寬度/高度可能為負數，要用 `max(0, ...)`
2. **NMS 無限迴圈** → 確保每次迴圈都移除至少一個框
3. **座標格式混淆** → 統一使用 `[x1, y1, x2, y2]` 格式
4. **除以零錯誤** → IoU 計算時 union 可能為 0，要檢查
5. **本 Lab 不需完整訓練** → 重點在演算法理解，不是訓練高準確率模型

## 🎯 預期成果

完成本作業後，你應該能夠：

- ✅ 正確計算任意兩個 Bounding Box 的 IoU
- ✅ 實作 NMS 演算法移除重複偵測
- ✅ 理解物件偵測與分類任務的差異
- ✅ 認識 mAP 評估指標的概念
- ✅ 知道如何使用 torchvision 的預訓練偵測模型

## 🤔 思考問題

1. 為什麼物件偵測比影像分類困難？需要解決哪些額外的問題？
2. IoU 閾值設為 0.5 代表什麼意義？改成 0.3 或 0.7 會有什麼影響？
3. NMS 的閾值如何影響最終結果？設太高或太低會怎樣？
4. 為何需要 mAP 而不是簡單的準確率？物件偵測的評估為何更複雜？
5. 如果一張影像中有 3 個物體，模型可能產生多少個偵測框？
6. 實際應用時，你會選擇從零訓練還是使用預訓練模型？為什麼？

## 📖 參考資源

### 核心論文
- [R-CNN](https://arxiv.org/abs/1311.2524) - 開創性的物件偵測方法
- [Fast R-CNN](https://arxiv.org/abs/1504.08083) - 改進版本
- [Faster R-CNN](https://arxiv.org/abs/1506.01497) - 當前主流方法
- [YOLO](https://arxiv.org/abs/1506.02640) - 即時偵測方法

### 教學資源
- [PyTorch Detection Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [PASCAL VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
- [Understanding IoU](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)
- [NMS Explained](https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/)

## 💡 進階學習方向

完成本 Lab 後，建議的進階方向：

1. **使用預訓練模型**
```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
predictions = model(images)
```

2. **Fine-tune 在自己的資料集**
   - 準備標註資料（需要 Bounding Box）
   - 使用遷移學習微調模型

3. **嘗試不同架構**
   - YOLO (更快，適合即時應用)
   - RetinaNet (處理小物體更好)
   - EfficientDet (效率與準確率平衡)

## ✅ 作業提交檢查清單

- [ ] 實作 `calculate_iou()` 並通過測試
- [ ] 實作 `non_maximum_suppression()` 並通過測試
- [ ] 理解 mAP 的計算概念
- [ ] 能夠解釋 IoU 和 NMS 的作用
- [ ] 知道物件偵測與分類的差異
- [ ] 回答所有思考問題
- [ ] （選做）嘗試使用 torchvision 的預訓練模型

---

**祝學習順利！🎉**

**重要提醒**: 本 Lab 著重**概念理解**，完整的物件偵測系統非常複雜。實際應用建議使用成熟的框架（如 torchvision, Detectron2）。
