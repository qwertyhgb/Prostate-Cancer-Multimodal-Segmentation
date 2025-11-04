"""
测试处理缺失模态数据的脚本

该脚本用于测试项目中处理缺失模态数据的不同策略，
包括数据统计、策略验证和功能测试。

测试内容:
- 缺失模态处理策略验证 (zero_fill, skip)
- 各模态数据数量统计
- 标签文件完整性检查
- 数据加载器功能测试

使用方式:
    python test_missing_modalities.py

作者: [项目作者]
版本: 1.0
"""

import os
import sys
import torch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script.data_loader import get_dataloader


def test_missing_modality_handling():
    """测试不同缺失模态处理策略
    
    验证项目中实现的缺失模态处理策略是否正常工作，
    包括数据加载、批次处理和策略效果评估。
    
    测试策略:
    - zero_fill: 用零填充缺失的模态数据
    - skip: 跳过包含缺失模态的病例
    
    测试流程:
    1. 为每种策略创建数据加载器
    2. 验证数据加载是否成功
    3. 检查批次数据的形状和完整性
    4. 统计有效病例数量
    """
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
    """统计各模态的数据数量
    
    统计每个模态目录下的图像文件数量，
    帮助了解数据集的完整性和分布情况。
    
    统计内容:
    - ADC模态: 弥散加权成像的表观扩散系数
    - DWI模态: 弥散加权成像
    - gaoqing-T2模态: 高分辨率T2加权成像
    - T2 fs模态: 脂肪抑制T2加权成像
    - T2 not fs模态: 非脂肪抑制T2加权成像
    
    输出:
        每个模态的文件数量和状态
    """
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
    """检查标签文件完整性
    
    验证标签目录是否存在，并统计标签文件数量，
    确保分割任务有对应的标注数据。
    
    检查内容:
    - 标签目录是否存在
    - 标签文件数量
    - 文件格式验证 (.nii格式)
    
    注意:
        标签文件应位于 ROI(BPH+PCA) 目录下，
        分别对应BPH和PCA的分割标注。
    """
    print("\n检查标签文件...")
    label_dir = 'data/BPH-PCA/ROI(BPH+PCA)/BPH'  # 更新为新的路径（英文括号）
    if os.path.exists(label_dir):
        count = len([f for f in os.listdir(label_dir) if f.endswith('.nii')])
        print(f"  标签文件: {count} 例")
    else:
        print("  标签目录不存在")


if __name__ == '__main__':
    """主函数 - 执行完整的缺失模态测试流程
    
    测试流程:
    1. 统计各模态数据数量
    2. 检查标签文件完整性
    3. 测试缺失模态处理策略
    4. 输出测试总结
    """
    # 统计模态数据数量
    count_modalities()
    
    # 检查标签文件
    check_labels()
    
    # 测试缺失模态处理策略
    test_missing_modality_handling()
    
    print("\n测试完成!")