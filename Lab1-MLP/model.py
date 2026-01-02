"""
模型架構模組 (Model Architecture Module)
定義 MLP (Multi-Layer Perceptron) 模型用於 MNIST 手寫數字分類

學生任務:
- 實作 __init__ 方法來建立網路層
- 實作 forward 方法來定義前向傳播
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    多層感知機 (Multi-Layer Perceptron) 模型
    
    架構:
    - 輸入層: 784 維度 (28x28 展平後的影像)
    - 隱藏層 1: 512 個神經元
    - 隱藏層 2: 256 個神經元  
    - 隱藏層 3: 128 個神經元
    - 輸出層: 10 個類別 (數字 0-9)
    
    每個隱藏層包含:
    - Linear 轉換
    - Batch Normalization
    - ReLU 激活函數
    - Dropout 正規化
    """
    
    def __init__(self, input_size=784, num_classes=10, dropout_rate=0.3):
        """
        初始化 MLP 模型
        
        Args:
            input_size (int): 輸入特徵維度 (28*28=784)
            num_classes (int): 輸出類別數量 (數字 0-9 共 10 類)
            dropout_rate (float): Dropout 機率 (0.3 = 30% dropout)
        """
        super(MLP, self).__init__()
        
        # ========================================
        # TODO: 學生實作區
        # ========================================
        # 請實作以下結構的神經網路:
        #
        # 隱藏層 1: 784 -> 512
        #   - Linear 層: nn.Linear(input_size, 512)
        #   - Batch normalization: nn.BatchNorm1d(512)
        #   - 激活函數: nn.ReLU()
        #   - Dropout: nn.Dropout(dropout_rate)
        #
        # 隱藏層 2: 512 -> 256
        #   - Linear 層: nn.Linear(512, 256)
        #   - Batch normalization: nn.BatchNorm1d(256)
        #   - 激活函數: nn.ReLU()
        #   - Dropout: nn.Dropout(dropout_rate)
        #
        # 隱藏層 3: 256 -> 128
        #   - Linear 層: nn.Linear(256, 128)
        #   - Batch normalization: nn.BatchNorm1d(128)
        #   - 激活函數: nn.ReLU()
        #   - Dropout: nn.Dropout(dropout_rate)
        #
        # 輸出層: 128 -> 10
        #   - Linear 層: nn.Linear(128, num_classes)
        #
        # 小提示:
        # - 可以使用 nn.Sequential() 將多層組合在一起
        # - BatchNorm 有助於穩定和加速訓練
        # - ReLU 是激活函數: f(x) = max(0, x)
        # - Dropout 會在訓練時隨機將神經元設為 0，防止過擬合
        # ========================================
        
        # 範例架構 (你需要填入細節):
        self.fc1 = None  # TODO: 第一個隱藏層 (784 -> 512)
        self.bn1 = None  # TODO: 第一層的 Batch normalization
        self.relu1 = None  # TODO: ReLU 激活函數
        self.dropout1 = None  # TODO: Dropout 層
        
        self.fc2 = None  # TODO: 第二個隱藏層 (512 -> 256)
        self.bn2 = None  # TODO: 第二層的 Batch normalization
        self.relu2 = None  # TODO: ReLU 激活函數
        self.dropout2 = None  # TODO: Dropout 層
        
        self.fc3 = None  # TODO: 第三個隱藏層 (256 -> 128)
        self.bn3 = None  # TODO: 第三層的 Batch normalization
        self.relu3 = None  # TODO: ReLU 激活函數
        self.dropout3 = None  # TODO: Dropout 層
        
        self.fc_out = None  # TODO: 輸出層 (128 -> 10)
        
    def forward(self, x):
        """
        前向傳播
        
        Args:
            x (torch.Tensor): 輸入張量，形狀為 (batch_size, 1, 28, 28)
        
        Returns:
            torch.Tensor: 輸出 logits，形狀為 (batch_size, 10)
        
        Tensor 形狀轉換:
            輸入:          (batch_size, 1, 28, 28)
            展平後:        (batch_size, 784)
            經過 fc1:      (batch_size, 512)
            經過 fc2:      (batch_size, 256)
            經過 fc3:      (batch_size, 128)
            輸出:          (batch_size, 10)
        """
        # ========================================
        # TODO: 學生實作區
        # ========================================
        # 請實作前向傳播，包含以下步驟:
        #
        # 步驟 1: 展平輸入
        #   - 輸入形狀: (batch_size, 1, 28, 28)
        #   - 輸出形狀: (batch_size, 784)
        #   - 提示: 使用 x.view(batch_size, -1) 或 x.flatten(1)
        #
        # 步驟 2: 通過隱藏層 1
        #   - 依序套用 fc1, bn1, relu1, dropout1
        #   - 形狀: (batch_size, 784) -> (batch_size, 512)
        #
        # 步驟 3: 通過隱藏層 2
        #   - 依序套用 fc2, bn2, relu2, dropout2
        #   - 形狀: (batch_size, 512) -> (batch_size, 256)
        #
        # 步驟 4: 通過隱藏層 3
        #   - 依序套用 fc3, bn3, relu3, dropout3
        #   - 形狀: (batch_size, 256) -> (batch_size, 128)
        #
        # 步驟 5: 通過輸出層
        #   - 套用 fc_out
        #   - 形狀: (batch_size, 128) -> (batch_size, 10)
        #   - 注意: 不要在這裡套用 softmax! CrossEntropyLoss 會自動處理
        #
        # 範例結構:
        # x = x.view(x.size(0), -1)  # 展平
        # x = self.fc1(x)
        # x = self.bn1(x)
        # x = self.relu1(x)
        # x = self.dropout1(x)
        # ... 繼續其他層 ...
        # ========================================
        
        raise NotImplementedError("學生需要實作前向傳播")


def create_mlp_model():
    """
    建立 MLP 模型實例
    
    Returns:
        MLP: 初始化的 MLP 模型
    """
    model = MLP(input_size=784, num_classes=10, dropout_rate=0.3)
    
    # 計算參數量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'✅ MLP 模型建立成功!')
    print(f'   架構: 784 -> 512 -> 256 -> 128 -> 10')
    print(f'   總參數量: {num_params:,}')
    
    return model


if __name__ == '__main__':
    # 測試模型
    print('🧪 測試 MLP 模型...\n')
    
    try:
        model = create_mlp_model()
        
        # 建立測試輸入
        batch_size = 4
        test_input = torch.randn(batch_size, 1, 28, 28)
        
        # 前向傳播
        output = model(test_input)
        
        print(f'\n📊 模型測試結果:')
        print(f'   輸入形狀: {test_input.shape}')
        print(f'   輸出形狀: {output.shape}')
        print(f'   預期輸出形狀: ({batch_size}, 10)')
        
        # 驗證輸出形狀
        assert output.shape == (batch_size, 10), '輸出形狀不正確!'
        print(f'\n✅ 模型測試通過!')
        
    except NotImplementedError:
        print('\n⚠️  模型尚未實作。學生需要完成 TODO 區塊。')
