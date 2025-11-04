#!/usr/bin/env python3
"""
前列腺多模态MRI分割项目 - 主运行脚本

这个脚本提供了一个统一的接口来运行项目的各种功能，包括训练、验证和预测。
"""

import argparse
import os
import sys
import torch
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def train_model(args):
    """训练模型"""
    print(f"开始训练模型...")
    print(f"使用配置: {args.config}")
    print(f"数据类型: {args.data_type}")
    
    if args.cross_validation:
        print("使用交叉验证训练...")
        if args.config == 'optimized':
            from train_bph_optimized import CrossValidationTrainer
            # 这里可以添加配置加载逻辑
            config = {
                'data_dir': args.data_dir,
                'data_type': args.data_type,
                'num_epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'device': args.device,
                'save_dir': args.save_dir,
                'n_splits': args.n_splits,
                'handle_missing_modalities': args.handle_missing
            }
            trainer = CrossValidationTrainer(config)
            # 这里需要添加实际的训练逻辑
        else:
            print("使用基础交叉验证脚本...")
            os.system(f"python train_bph_cv.py")
    else:
        print("使用标准训练...")
        if args.config == 'optimized':
            from train_bph_optimized import BPHTrainer
            config = {
                'data_dir': args.data_dir,
                'data_type': args.data_type,
                'num_epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'device': args.device,
                'save_dir': args.save_dir,
                'handle_missing_modalities': args.handle_missing
            }
            trainer = BPHTrainer(config)
            # 这里需要添加实际的训练逻辑
        else:
            print("使用基础训练脚本...")
            os.system(f"python train_bph.py")

def validate_model(args):
    """验证模型"""
    print(f"开始验证模型...")
    print(f"模型路径: {args.model_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件 {args.model_path} 不存在")
        return
    
    # 运行验证脚本
    cmd = f"python script/validate_model.py --model_path {args.model_path}"
    if args.data_dir:
        cmd += f" --data_dir {args.data_dir}"
    if args.data_type:
        cmd += f" --data_type {args.data_type}"
    
    os.system(cmd)

def predict_data(args):
    """预测新数据"""
    print(f"开始预测数据...")
    print(f"模型路径: {args.model_path}")
    print(f"输入数据: {args.input_data}")
    print(f"输出路径: {args.output_path}")
    
    # 检查必要文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件 {args.model_path} 不存在")
        return
    
    if not os.path.exists(args.input_data):
        print(f"错误: 输入数据 {args.input_data} 不存在")
        return
    
    # 运行预测脚本
    cmd = f"python script/predict.py --model_path {args.model_path} --input_data {args.input_data} --output_path {args.output_path}"
    os.system(cmd)

def check_environment():
    """检查运行环境"""
    print("检查运行环境...")
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查PyTorch
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            print(f"当前GPU: {torch.cuda.current_device()}")
    except ImportError:
        print("错误: 未安装PyTorch")
        return False
    
    # 检查其他依赖
    dependencies = ['tqdm', 'SimpleITK', 'numpy', 'scikit-learn']
    missing_deps = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep} 已安装")
        except ImportError:
            missing_deps.append(dep)
            print(f"✗ {dep} 未安装")
    
    if missing_deps:
        print(f"\n请安装缺失的依赖: pip install {' '.join(missing_deps)}")
        return False
    
    print("环境检查完成 ✓")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='前列腺多模态MRI分割项目 - 主运行脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 训练模型
  python run.py train --config optimized --data_type BPH
  
  # 交叉验证训练
  python run.py train --cross_validation --config optimized
  
  # 验证模型
  python run.py validate --model_path checkpoints/best_model.pth
  
  # 预测新数据
  python run.py predict --model_path checkpoints/best_model.pth --input_data data/test_case --output_path results/
  
  # 检查环境
  python run.py check
        """
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', choices=['standard', 'optimized'], 
                            default='optimized', help='训练配置类型')
    train_parser.add_argument('--data_type', choices=['BPH', 'PCA'], 
                            default='BPH', help='数据类型')
    train_parser.add_argument('--data_dir', default='data', help='数据目录')
    train_parser.add_argument('--save_dir', default='checkpoints', help='模型保存目录')
    train_parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    train_parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    train_parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    train_parser.add_argument('--device', default='auto', help='训练设备')
    train_parser.add_argument('--cross_validation', action='store_true', 
                            help='使用交叉验证')
    train_parser.add_argument('--n_splits', type=int, default=5, help='交叉验证折数')
    train_parser.add_argument('--handle_missing', 
                            choices=['zero_fill', 'skip', 'duplicate'],
                            default='zero_fill', help='缺失模态处理策略')
    
    # 验证命令
    validate_parser = subparsers.add_parser('validate', help='验证模型')
    validate_parser.add_argument('--model_path', required=True, help='模型文件路径')
    validate_parser.add_argument('--data_dir', default='data', help='数据目录')
    validate_parser.add_argument('--data_type', choices=['BPH', 'PCA'], 
                               help='数据类型')
    
    # 预测命令
    predict_parser = subparsers.add_parser('predict', help='预测新数据')
    predict_parser.add_argument('--model_path', required=True, help='模型文件路径')
    predict_parser.add_argument('--input_data', required=True, help='输入数据路径')
    predict_parser.add_argument('--output_path', required=True, help='输出结果路径')
    
    # 环境检查命令
    check_parser = subparsers.add_parser('check', help='检查运行环境')
    
    args = parser.parse_args()
    
    # 自动检测设备
    if hasattr(args, 'device') and args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"自动选择设备: {args.device}")
    
    # 执行命令
    if args.command == 'train':
        train_model(args)
    elif args.command == 'validate':
        validate_model(args)
    elif args.command == 'predict':
        predict_data(args)
    elif args.command == 'check':
        check_environment()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()