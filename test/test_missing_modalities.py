"""
测试处理缺失模态数据的脚本
"""

import os
import sys
import torch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script.data_loader import get_dataloader

def test_missing_modality_handling():
    """测试不同缺失模态处理策略"""
    print("测试缺失模态处理策略...")
    
    # 测试配置
    config = {
        'data_dir': 'data',
        'batch_size': 2,
        'data_type': 'BPH'
    }
    
    # 策略列表
    strategies = ['zero_fill', 'skip']
    
    for strategy in strategies:
        print(f"\n测试策略: {strategy}")
        try:
            # 创建数据加载器
            dataloader = get_dataloader(
                root_dir=config['data_dir'],
                batch_size=config['batch_size'],
                mode='train',
                data_type=config['data_type'],
                handle_missing_modalities=strategy,
                num_workers=0  # 避免Windows上的多进程问题
            )
            
            print(f"  数据集大小: {len(dataloader.dataset)}")
            
            # 尝试获取一个批次的数据
            batch = next(iter(dataloader))
            print(f"  成功加载数据，批次大小: {batch['image'].shape[0]}")
            print(f"  图像形状: {batch['image'].shape}")
            print(f"  标签形状: {batch['label'].shape}")
            
            # 检查是否有缺失的病例ID
            if strategy == 'skip':
                print(f"  使用完整数据的病例数: {len(dataloader.dataset.cases)}")
                
        except Exception as e:
            print(f"  错误: {e}")

def count_modalities():
    """统计各模态的数据数量"""
    print("\n统计各模态数据数量...")
    
    modalities = ['ADC', 'DWI', 'gaoqing-T2', 'T2 fs', 'T2 not fs']
    data_dir = 'data/BPH-PCA/BPH'
    
    for modality in modalities:
        modality_dir = os.path.join(data_dir, modality)
        if os.path.exists(modality_dir):
            count = len([f for f in os.listdir(modality_dir) if f.endswith('.nii')])
            print(f"  {modality}: {count} 例")
        else:
            print(f"  {modality}: 目录不存在")

def check_labels():
    """检查标签文件"""
    print("\n检查标签文件...")
    label_dir = 'data/BPH-PCA/ROI(BPH+PCA)/BPH'  # 更新为新的路径（英文括号）
    if os.path.exists(label_dir):
        count = len([f for f in os.listdir(label_dir) if f.endswith('.nii')])
        print(f"  标签文件: {count} 例")
    else:
        print("  标签目录不存在")

if __name__ == '__main__':
    # 统计模态数据数量
    count_modalities()
    
    # 检查标签文件
    check_labels()
    
    # 测试缺失模态处理策略
    test_missing_modality_handling()
    
    print("\n测试完成!")