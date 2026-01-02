"""
準備 MNIST 資料集為 JSON 格式
此腳本下載 MNIST 並將其轉換為 JSON + PNG 格式供作業使用

執行此腳本一次以準備資料集:
    python prepare_mnist.py
"""

import torch
from torchvision import datasets
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm


def prepare_mnist_json(data_dir='./data'):
    """
    下載 MNIST 並轉換為 JSON 格式
    
    建立:
        data/train.json - 訓練集標註
        data/test.json - 測試集標註
        data/train/*.png - 訓練影像
        data/test/*.png - 測試影像
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    print('📥 下載 MNIST 資料集...')
    
    # 使用 torchvision 下載 MNIST
    train_dataset = datasets.MNIST(root=str(data_dir), train=True, download=True)
    test_dataset = datasets.MNIST(root=str(data_dir), train=False, download=True)
    
    # 建立影像目錄
    train_img_dir = data_dir / 'train'
    test_img_dir = data_dir / 'test'
    train_img_dir.mkdir(exist_ok=True)
    test_img_dir.mkdir(exist_ok=True)
    
    # 處理訓練集
    print('\n📝 轉換訓練集為 JSON 格式...')
    train_samples = []
    for idx, (img, label) in enumerate(tqdm(train_dataset)):
        img_id = f'{idx:05d}'
        img_path = f'train/{img_id}.png'
        
        # 儲存影像
        img.save(data_dir / img_path)
        
        # 加入樣本列表
        train_samples.append({
            'id': img_id,
            'image_path': img_path,
            'label': int(label)
        })
    
    # 儲存訓練 JSON
    train_json = {
        'num_samples': len(train_samples),
        'num_classes': 10,
        'samples': train_samples
    }
    with open(data_dir / 'train.json', 'w') as f:
        json.dump(train_json, f, indent=2)
    
    print(f'✅ 已儲存 {len(train_samples)} 個訓練樣本')
    
    # 處理測試集
    print('\n📝 轉換測試集為 JSON 格式...')
    test_samples = []
    for idx, (img, label) in enumerate(tqdm(test_dataset)):
        img_id = f'{idx:05d}'
        img_path = f'test/{img_id}.png'
        
        # 儲存影像
        img.save(data_dir / img_path)
        
        # 加入樣本列表
        test_samples.append({
            'id': img_id,
            'image_path': img_path,
            'label': int(label)
        })
    
    # 儲存測試 JSON
    test_json = {
        'num_samples': len(test_samples),
        'num_classes': 10,
        'samples': test_samples
    }
    with open(data_dir / 'test.json', 'w') as f:
        json.dump(test_json, f, indent=2)
    
    print(f'✅ 已儲存 {len(test_samples)} 個測試樣本')
    
    print('\n🎉 資料集準備完成！')
    print(f'\n資料集結構:')
    print(f'  {data_dir}/train.json ({len(train_samples)} 個樣本)')
    print(f'  {data_dir}/test.json ({len(test_samples)} 個樣本)')
    print(f'  {data_dir}/train/*.png (訓練影像)')
    print(f'  {data_dir}/test/*.png (測試影像)')
    
    # 印出範例 JSON 項目
    print(f'\nJSON 項目範例:')
    print(json.dumps(train_samples[0], indent=2))


if __name__ == '__main__':
    prepare_mnist_json()
