"""
模型验证脚本 - 用于评估训练好的3D U-Net模型在测试集上的性能

该脚本提供完整的模型验证流程，包括模型加载、测试数据评估、性能指标计算和结果保存。
支持Dice系数和IoU等医学图像分割常用评估指标的计算和统计。

作者: [项目作者]
版本: 1.0
创建时间: [项目创建时间]
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from models.unet3d import UNet3D
from utils.losses import DiceLoss
from script.data_loader import get_dataloader
import SimpleITK as sitk
from datetime import datetime
import json


def calculate_dice_score(pred, target):
    """计算Dice系数
    
    Dice系数是医学图像分割中最常用的评估指标之一，
    用于衡量预测分割结果与真实标签之间的相似度。
    
    数学公式:
        Dice = (2 * |X ∩ Y|) / (|X| + |Y|)
    
    参数:
        pred (torch.Tensor): 预测结果，二值分割图（0或1）
        target (torch.Tensor): 真实标签，二值分割图（0或1）
        
    返回:
        float: Dice系数，范围[0,1]，值越大表示分割效果越好
        
    特性:
    - 对类别不平衡不敏感
    - 直接评估分割边界质量
    - 值域为[0,1]，1表示完美分割
    """
    # 展平张量，将多维张量转换为一维向量
    # 便于计算交集和并集
    pred = pred.view(-1)
    target = target.view(-1)
    
    # 计算交集（预测值和真实值的元素级乘积之和）
    intersection = (pred * target).sum()
    
    # 计算Dice系数
    # 分子: 2倍交集
    # 分母: 预测值总和 + 真实值总和 + 平滑因子（防止除零）
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-8)
    return dice.item()


def calculate_iou(pred, target):
    """计算IoU (Intersection over Union)
    
    IoU也称为Jaccard指数，用于衡量两个样本的重叠程度，
    在目标检测和图像分割任务中广泛使用。
    
    数学公式:
        IoU = |X ∩ Y| / |X ∪ Y|
    
    参数:
        pred (torch.Tensor): 预测结果，二值分割图（0或1）
        target (torch.Tensor): 真实标签，二值分割图（0或1）
        
    返回:
        float: IoU值，范围[0,1]，值越大表示重叠度越高
        
    特性:
    - 衡量预测区域与真实区域的重叠比例
    - 对分割边界的精确度敏感
    - 值域为[0,1]，1表示完全重叠
    """
    # 展平张量，将多维张量转换为一维向量
    pred = pred.view(-1)
    target = target.view(-1)
    
    # 计算交集（预测值和真实值的元素级乘积之和）
    intersection = (pred * target).sum()
    
    # 计算并集（预测值总和 + 真实值总和 - 交集）
    union = pred.sum() + target.sum() - intersection
    
    # 计算IoU
    # 分子: 交集
    # 分母: 并集 + 平滑因子（防止除零）
    iou = intersection / (union + 1e-8)
    return iou.item()


class ModelValidator:
    """模型验证器
    
    专门用于评估训练好的3D U-Net模型在测试集上的性能，
    提供完整的验证流程管理和结果统计分析。
    
    功能特性:
    - 自动加载训练好的模型权重
    - 批量处理测试数据
    - 计算多种评估指标（Dice系数、IoU等）
    - 保存详细的验证结果
    - 支持病例级别的性能分析
    """
    
    def __init__(self, config):
        """
        初始化模型验证器
        
        参数:
            config (dict): 验证配置字典，包含以下键值:
                - model_path (str): 模型文件路径（必需）
                - data_dir (str): 数据集根目录路径
                - batch_size (int): 批次大小
                - device (str): 验证设备 ('cuda' 或 'cpu')
                - data_type (str): 数据类型 ('BPH' 或 'PCA')
                - save_dir (str): 结果保存目录
                - handle_missing_modalities (str): 处理缺失模态的方法
        
        初始化流程:
        1. 设置验证设备和配置参数
        2. 加载训练好的模型
        3. 创建测试数据加载器
        4. 创建结果保存目录
        """
        self.config = config
        # 设置设备（GPU或CPU）
        self.device = torch.device(config['device'])
        
        # 加载模型
        self.model = self._load_model()
        
        # 获取处理缺失模态的策略
        handle_missing = config.get('handle_missing_modalities', 'zero_fill')
        
        # 创建测试数据加载器
        self.test_loader = get_dataloader(
            config['data_dir'],
            batch_size=config['batch_size'],
            mode='test',
            data_type=config.get('data_type', 'BPH'),
            handle_missing_modalities=handle_missing
        )
        
        # 创建保存目录
        os.makedirs(config['save_dir'], exist_ok=True)
        
    def _load_model(self):
        """加载训练好的模型
        
        返回:
            UNet3D: 加载完成的模型实例，设置为评估模式
            
        加载流程:
        1. 创建模型架构实例
        2. 从检查点文件加载权重
        3. 设置模型为评估模式
        4. 返回配置好的模型
        """
        print("正在加载模型...")
        # 创建模型实例
        model = UNet3D(
            n_modalities=5,  # 5个模态: ADC, DWI, T2 fs, T2 not fs, gaoqing-T2
            n_classes=1      # 二分类问题：背景和前列腺病变
        ).to(self.device)
        
        # 加载模型权重
        checkpoint = torch.load(self.config['model_path'], map_location=self.device)
        if 'model_state_dict' in checkpoint:
            # 如果检查点包含模型状态字典（完整检查点格式）
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果检查点直接是模型状态字典（仅模型权重格式）
            model.load_state_dict(checkpoint)
            
        # 设置为评估模式
        # 禁用dropout和batch normalization的随机性
        model.eval()
        print("模型加载完成!")
        return model
        
    def validate(self):
        """验证模型性能
        
        在测试集上评估模型性能，计算Dice系数和IoU等指标，
        并保存详细的验证结果。
        
        返回:
            tuple: (平均Dice系数, 平均IoU)
            
        验证流程:
        1. 初始化统计变量和结果存储
        2. 遍历测试数据加载器
        3. 对每个批次进行模型预测
        4. 计算每个病例的评估指标
        5. 统计平均性能指标
        6. 保存详细结果到JSON文件
        """
        print("开始模型验证...")
        
        # 初始化统计变量
        total_dice = 0
        total_iou = 0
        count = 0
        
        # 存储每个病例的结果
        results = []
        
        # 关闭梯度计算以节省内存和计算资源
        with torch.no_grad():
            # 遍历测试数据，使用进度条显示验证进度
            for batch in tqdm(self.test_loader, desc='验证模型'):
                # 将数据移动到指定设备
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                case_ids = batch['case_id']
                
                # 模型预测
                # 使用predict方法获得概率图，然后二值化
                outputs = self.model.predict(images)
                
                # 计算评估指标
                for i in range(outputs.shape[0]):
                    # 二值化预测结果（阈值0.5）
                    pred = (outputs[i] > 0.5).float()
                    
                    # 计算Dice系数和IoU
                    dice = calculate_dice_score(pred, labels[i])
                    iou = calculate_iou(pred, labels[i])
                    
                    # 累加统计值
                    total_dice += dice
                    total_iou += iou
                    count += 1
                    
                    # 保存每个病例的结果
                    case_result = {
                        'case_id': case_ids[i],
                        'dice': dice,
                        'iou': iou
                    }
                    results.append(case_result)
        
        # 计算平均指标
        avg_dice = total_dice / count
        avg_iou = total_iou / count
        
        # 打印结果
        print(f"\n验证结果:")
        print(f"  平均Dice系数: {avg_dice:.4f}")
        print(f"  平均IoU: {avg_iou:.4f}")
        print(f"  验证病例数: {count}")
        
        # 保存结果到JSON文件
        results_dict = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'avg_dice': avg_dice,
            'avg_iou': avg_iou,
            'case_count': count,
            'case_results': results
        }
        
        results_path = os.path.join(self.config['save_dir'], 'validation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
            
        print(f"详细结果已保存至: {results_path}")
        return avg_dice, avg_iou


def main():
    """主函数 - 配置和启动模型验证流程
    
    功能:
    - 设置验证配置参数
    - 检查模型文件有效性
    - 创建验证器实例
    - 执行验证流程
    
    使用说明:
    1. 修改config['model_path']为实际的模型文件路径
    2. 根据需要调整其他配置参数
    3. 运行脚本进行模型验证
    """
    # 验证配置参数
    config = {
        'model_path': '',  # 需要指定模型文件路径（必需）
        'data_dir': 'data',
        'batch_size': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data_type': 'BPH',
        'handle_missing_modalities': 'zero_fill',  # 处理缺失模态的方法
        'save_dir': os.path.join('results', f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    }
    
    # 检查模型文件路径是否已设置
    if not config['model_path'] or not os.path.exists(config['model_path']):
        print("请在config中指定有效的模型文件路径!")
        print("例如: config['model_path'] = 'checkpoints/BPH_20240101_120000/best_model_epoch_50.pth'")
        return
    
    # 创建验证器并执行验证
    validator = ModelValidator(config)
    validator.validate()


if __name__ == '__main__':
    main()