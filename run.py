#!/usr/bin/env python3
"""
前列腺多模态MRI分割项目 - 主运行脚本

功能概述：
- 提供统一的命令行接口，支持训练、验证、预测和环境检查
- 支持多种配置选项和运行模式
- 集成交叉验证和缺失模态处理功能
- 提供详细的日志输出和进度监控

使用示例：
1. 环境检查: python run.py check
2. 训练模型: python run.py train --data_type BPH --epochs 100
3. 验证模型: python run.py validate --model_path models/best_model.pth
4. 预测分割: python run.py predict --input_dir data/test --output_dir results

作者: 项目团队
版本: 1.0
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from script.data_loader import get_dataloader, get_kfold_splits
from models.unet3d import UNet3D
from utils.trainer import Trainer

def check_environment():
    """
    检查运行环境是否满足要求
    
    验证内容：
    - PyTorch版本和CUDA可用性
    - 必要的Python包是否安装
    - GPU内存是否充足
    - 数据目录是否存在
    
    返回:
        bool: 环境检查是否通过
    """
    print("=" * 60)
    print("开始环境检查...")
    print("=" * 60)
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA可用: 是")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA可用: 否 (将使用CPU)")
    
    # 检查必要的包
    required_packages = ['numpy', 'SimpleITK', 'scikit-learn', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"{package}: 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"{package}: 未安装")
    
    # 检查数据目录
    data_dirs = ['data/BPH-PCA/BPH', 'data/BPH-PCA/PCA']
    missing_dirs = []
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            print(f"数据目录 {data_dir}: 存在")
        else:
            missing_dirs.append(data_dir)
            print(f"数据目录 {data_dir}: 不存在")
    
    # 输出检查结果
    print("-" * 60)
    
    if missing_packages:
        print(f"警告: 缺少必要的包: {missing_packages}")
        print("请运行: pip install " + " ".join(missing_packages))
    
    if missing_dirs:
        print(f"警告: 缺少数据目录: {missing_dirs}")
        print("请确保数据文件已正确放置")
    
    if not missing_packages and not missing_dirs:
        print("✓ 环境检查通过!")
        return True
    else:
        print("✗ 环境检查未通过，请解决上述问题")
        return False

def train_model(args):
    """
    训练3D U-Net模型
    
    支持功能：
    - 标准训练和优化训练模式
    - 交叉验证训练
    - 缺失模态处理
    - 早停机制和模型保存
    
    参数:
        args: 命令行参数对象
    """
    print("=" * 60)
    print("开始模型训练...")
    print(f"数据类型: {args.data_type}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = UNet3D(n_modalities=5, n_classes=2)
    model.to(device)
    
    # 选择训练模式
    if args.optimized:
        print("使用优化训练模式")
        from train_bph_optimized import train_optimized
        train_optimized(model, device, args)
    elif args.cross_validation:
        print("使用交叉验证训练模式")
        from train_bph_cv import train_with_cross_validation
        train_with_cross_validation(model, device, args)
    else:
        print("使用标准训练模式")
        from train_bph import train_standard
        train_standard(model, device, args)
    
    print("训练完成!")

def validate_model(args):
    """
    验证训练好的模型
    
    功能：
    - 加载预训练模型
    - 在验证集上评估性能
    - 计算Dice系数等指标
    - 生成性能报告
    
    参数:
        args: 命令行参数对象
    """
    print("=" * 60)
    print("开始模型验证...")
    print(f"模型路径: {args.model_path}")
    print("=" * 60)
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = UNet3D(n_modalities=5, n_classes=2)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"模型加载成功，使用设备: {device}")
    
    # 执行验证
    from script.validate_model import validate
    validate(model, device, args)
    
    print("验证完成!")

def predict_segmentation(args):
    """
    使用训练好的模型进行分割预测
    
    功能：
    - 加载预训练模型
    - 对输入图像进行分割
    - 保存分割结果
    - 支持批量预测
    
    参数:
        args: 命令行参数对象
    """
    print("=" * 60)
    print("开始分割预测...")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = UNet3D(n_modalities=5, n_classes=2)
    
    if args.model_path:
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"加载预训练模型: {args.model_path}")
        else:
            print(f"警告: 模型文件不存在，使用随机初始化模型")
    
    model.to(device)
    model.eval()
    
    print(f"使用设备: {device}")
    
    # 执行预测
    # 这里需要实现具体的预测逻辑
    # 由于预测功能较为复杂，建议在单独的模块中实现
    print("预测功能正在开发中...")
    print("请参考 script/predict.py 模块")
    
    print("预测完成!")

def main():
    """
    主函数 - 解析命令行参数并执行相应功能
    
    命令行参数说明：
    - check: 环境检查
    - train: 模型训练
    - validate: 模型验证
    - predict: 分割预测
    
    每个子命令都有特定的参数选项
    """
    # 创建主解析器
    parser = argparse.ArgumentParser(
        description='前列腺多模态MRI分割项目 - 主运行脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 环境检查
  python run.py check
  
  # 训练BPH数据模型
  python run.py train --data_type BPH --epochs 100 --batch_size 2
  
  # 使用交叉验证训练
  python run.py train --data_type BPH --cross_validation --epochs 50
  
  # 验证模型
  python run.py validate --model_path models/best_model.pth
  
  # 预测分割
  python run.py predict --input_dir data/test --output_dir results
        """
    )
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # check命令
    check_parser = subparsers.add_parser('check', help='检查运行环境')
    
    # train命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--data_type', type=str, default='BPH', 
                             choices=['BPH', 'PCA'], help='数据类型 (BPH或PCA)')
    train_parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    train_parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    train_parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    train_parser.add_argument('--optimized', action='store_true', help='使用优化训练')
    train_parser.add_argument('--cross_validation', action='store_true', 
                             help='使用交叉验证')
    train_parser.add_argument('--missing_strategy', type=str, default='zero_fill',
                             choices=['zero_fill', 'skip', 'duplicate'], 
                             help='缺失模态处理策略')
    
    # validate命令
    validate_parser = subparsers.add_parser('validate', help='验证模型')
    validate_parser.add_argument('--model_path', type=str, required=True,
                                help='模型文件路径')
    validate_parser.add_argument('--data_type', type=str, default='BPH',
                                choices=['BPH', 'PCA'], help='验证数据类型')
    validate_parser.add_argument('--batch_size', type=int, default=2, 
                                help='验证批次大小')
    
    # predict命令
    predict_parser = subparsers.add_parser('predict', help='预测分割')
    predict_parser.add_argument('--input_dir', type=str, required=True,
                               help='输入图像目录')
    predict_parser.add_argument('--output_dir', type=str, required=True,
                               help='输出结果目录')
    predict_parser.add_argument('--model_path', type=str, default='',
                               help='模型文件路径（可选）')
    predict_parser.add_argument('--batch_size', type=int, default=1,
                               help='预测批次大小')
    
    # 解析参数
    args = parser.parse_args()
    
    # 如果没有提供命令，显示帮助信息
    if not args.command:
        parser.print_help()
        return
    
    # 根据命令执行相应功能
    try:
        if args.command == 'check':
            check_environment()
        elif args.command == 'train':
            train_model(args)
        elif args.command == 'validate':
            validate_model(args)
        elif args.command == 'predict':
            predict_segmentation(args)
        else:
            print(f"未知命令: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"执行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    """
    脚本入口点
    
    当直接运行此脚本时，执行main函数
    当被导入为模块时，不执行main函数
    """
    main()