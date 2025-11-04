"""
优化训练脚本 - 提供多种训练模式选择的BPH数据训练器

该脚本提供两种训练模式：基础训练和交叉验证训练，
用户可以根据数据量和需求选择合适的训练策略。

功能特性:
- 基础训练模式：标准训练流程，适用于大数据集
- 交叉验证训练模式：K折交叉验证，适用于小数据集
- 交互式训练模式选择
- 自动模型保存和结果记录
- 早停机制防止过拟合

作者: [项目作者]
版本: 1.0
创建时间: [项目创建时间]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from models.unet3d import UNet3D
from utils.losses import DiceLoss
from utils.trainer import BaseTrainer
from script.data_loader import get_dataloader, get_kfold_splits
import os
from datetime import datetime
import json


class BPHTrainer(BaseTrainer):
    """BPH数据专用模型训练器
    
    继承自BaseTrainer，专门针对BPH数据进行优化，
    提供标准训练流程，适用于相对较大的数据集。
    
    使用场景:
    - 数据量较大时（如超过500例）
    - 需要快速训练和验证
    - 标准训练-验证集分割即可满足需求
    """
    
    def __init__(self, config):
        """
        初始化BPH训练器
        
        参数:
            config (dict): 训练配置字典，包含以下键值:
                - data_dir (str): 数据集根目录路径
                - num_epochs (int): 训练轮数
                - batch_size (int): 批次大小
                - learning_rate (float): 学习率
                - device (str): 训练设备 ('cuda' 或 'cpu')
                - save_dir (str): 模型保存目录
                - data_type (str): 数据类型 ('BPH' 或 'PCA')
                - handle_missing_modalities (str): 处理缺失模态的方法
        
        初始化流程:
        1. 确保数据类型为BPH
        2. 调用父类BaseTrainer的初始化方法
        3. 打印训练器信息和提示
        """
        # 确保数据类型为BPH
        config['data_type'] = 'BPH'
        super().__init__(config)
        
        print("="*60)
        print("BPH数据专用训练器")
        print("="*60)
        print("注意：对于小数据集（如240例），建议使用交叉验证版本")
        print("以更好地利用数据并获得更稳定的模型性能评估。")
        print("="*60)


class CrossValidationTrainer:
    """交叉验证训练器
    
    使用K折交叉验证进行模型训练和评估，
    特别适用于小数据集，能够充分利用有限的数据资源。
    
    交叉验证优势:
    - 每个样本都参与训练和验证
    - 提供更稳定的性能评估
    - 减少过拟合风险
    - 获得多个独立模型用于集成
    """
    
    def __init__(self, config):
        """
        初始化交叉验证训练器
        
        参数:
            config (dict): 训练配置字典，包含以下键值:
                - data_dir (str): 数据集根目录路径
                - num_epochs (int): 训练轮数
                - batch_size (int): 批次大小
                - learning_rate (float): 学习率
                - device (str): 训练设备 ('cuda' 或 'cpu')
                - save_dir (str): 模型保存目录
                - data_type (str): 数据类型 ('BPH' 或 'PCA')
                - n_splits (int): 交叉验证折数
                - handle_missing_modalities (str): 处理缺失模态的方法
        
        初始化流程:
        1. 设置训练设备和配置参数
        2. 获取K折交叉验证分割索引
        3. 初始化结果存储结构
        4. 创建模型保存目录
        """
        self.config = config
        self.device = torch.device(config['device'])
        self.n_splits = config.get('n_splits', 5)  # 默认5折交叉验证
        self.handle_missing = config.get('handle_missing_modalities', 'zero_fill')
        
        # 获取K折分割索引
        # 将数据集分割为K个不重叠的子集用于交叉验证
        self.kfold_splits = get_kfold_splits(
            config['data_dir'],
            data_type=config.get('data_type', 'BPH'),
            n_splits=self.n_splits,
            missing_strategy=self.handle_missing
        )
        
        # 记录每折的结果
        self.fold_results = []
        
        # 创建保存目录
        os.makedirs(config['save_dir'], exist_ok=True)
        
    def _create_model(self):
        """创建新的模型实例
        
        为每一折创建独立的模型实例，避免交叉验证中的数据泄露问题。
        
        返回:
            UNet3D: 新的3D U-Net模型实例
            
        模型配置:
        - 输入模态数: 5 (ADC, DWI, T2 fs, T2 not fs, gaoqing-T2)
        - 输出类别数: 1 (二分类问题：背景和前列腺病变)
        """
        return UNet3D(
            n_modalities=5,  # 5个模态: ADC, DWI, T2 fs, T2 not fs, gaoqing-T2
            n_classes=1      # 二分类问题：背景和前列腺病变
        ).to(self.device)
        
    def _create_optimizer(self, model):
        """为模型创建优化器
        
        使用Adam优化器，结合权重衰减以防止过拟合。
        
        参数:
            model (UNet3D): 模型实例
            
        返回:
            Adam: Adam优化器实例
            
        优化器配置:
        - 学习率: 从配置中获取
        - 权重衰减: 1e-5 (L2正则化)
        """
        return optim.Adam(
            model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=1e-5  # 添加权重衰减以防止过拟合
        )
        
    def _create_scheduler(self, optimizer):
        """为优化器创建学习率调度器
        
        使用ReduceLROnPlateau调度器，当验证损失停止改善时自动降低学习率。
        
        参数:
            optimizer (Adam): 优化器实例
            
        返回:
            ReduceLROnPlateau: 学习率调度器实例
            
        调度器配置:
        - 监控指标: 验证损失 (mode='min')
        - 耐心值: 10个epoch
        - 降低因子: 0.5
        """
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=10, 
            factor=0.5
        )
        
    def train_fold(self, fold_idx, train_indices, val_indices):
        """训练单个折
        
        对指定的训练集和验证集进行完整的模型训练和验证。
        
        参数:
            fold_idx (int): 当前折的索引 (0-indexed)
            train_indices (list): 训练集索引列表
            val_indices (list): 验证集索引列表
            
        返回:
            dict: 该折的训练结果，包含最佳验证损失等信息
            
        训练流程:
        1. 创建独立的模型、优化器和调度器
        2. 准备训练和验证数据加载器
        3. 执行多轮训练和验证
        4. 应用早停机制防止过拟合
        5. 保存最佳模型检查点
        """
        print(f"\n{'='*60}")
        print(f"开始训练第 {fold_idx + 1}/{self.n_splits} 折")
        print(f"训练集大小: {len(train_indices)}, 验证集大小: {len(val_indices)}")
        print(f"处理缺失模态的方法: {self.handle_missing}")
        print(f"{'='*60}")
        
        # 创建模型、优化器和调度器
        model = self._create_model()
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer)
        criterion = DiceLoss()
        
        # 创建数据加载器
        train_loader = get_dataloader(
            self.config['data_dir'],
            batch_size=self.config['batch_size'],
            data_type=self.config.get('data_type', 'BPH'),
            indices=train_indices,
            missing_strategy=self.handle_missing,
            is_training=True,
            num_workers=0  # 显式设置为0以避免在某些系统上出现警告
        )
        
        val_loader = get_dataloader(
            self.config['data_dir'],
            batch_size=self.config['batch_size'],
            data_type=self.config.get('data_type', 'BPH'),
            indices=val_indices,
            missing_strategy=self.handle_missing,
            is_training=False,
            num_workers=0  # 显式设置为0以避免在某些系统上出现警告
        )
        
        # 初始化混合精度训练的梯度缩放器
        scaler = GradScaler()
        
        # 训练循环
        best_val_loss = float('inf')  # 初始化最佳验证损失
        patience_counter = 0  # 早停计数器
        max_patience = 15  # 最大耐心值（连续15个epoch无改善则停止）
        
        for epoch in range(self.config['num_epochs']):
            # 训练阶段
            model.train()
            train_loss = 0
            
            with tqdm(train_loader, desc=f'Fold {fold_idx+1} Training') as pbar:
                for batch in pbar:
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    # 前向传播和反向传播（使用混合精度）
                    optimizer.zero_grad()
                    
                    # 使用自动混合精度
                    with autocast():
                        outputs = model(images)
                        
                        # 确保输出和标签的形状匹配
                        if outputs.shape != labels.shape:
                            # 调整标签形状以匹配输出
                            if len(labels.shape) != len(outputs.shape):
                                # 如果标签缺少通道维度，则添加
                                if len(labels.shape) == 4 and len(outputs.shape) == 5:
                                    labels = labels.unsqueeze(1)  # 添加通道维度
                                elif len(labels.shape) == 5 and len(outputs.shape) == 4:
                                    outputs = outputs.unsqueeze(1)  # 添加通道维度
                            else:
                                # 如果维度数量相同但尺寸不同，尝试调整标签以匹配输出
                                # 这里我们假设输出的尺寸是模型期望的尺寸
                                # 我们需要将标签调整为与输出相同的尺寸
                                # 使用最近邻插值方法调整标签尺寸，确保结果仍然是二值的
                                if labels.shape[2:] != outputs.shape[2:]:
                                    # 只调整空间维度
                                    # 保存原始标签的数据类型
                                    original_dtype = labels.dtype
                                    # 转换为浮点型进行插值，然后转换回原始类型
                                    labels = F.interpolate(labels.float(), size=outputs.shape[2:], mode='nearest').to(original_dtype)
                        
                        loss = criterion(outputs, labels)
                    
                    # 缩放损失并反向传播
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
            avg_train_loss = train_loss / len(train_loader)
            
            # 验证阶段
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                with tqdm(val_loader, desc=f'Fold {fold_idx+1} Validation') as pbar:
                    for batch in pbar:
                        images = batch['image'].to(self.device)
                        labels = batch['label'].to(self.device)
                        
                        # 使用自动混合精度进行推理
                        with autocast():
                            outputs = model(images)
                            
                            # 确保输出和标签的形状匹配
                            if outputs.shape != labels.shape:
                                # 调整标签形状以匹配输出
                                if len(labels.shape) != len(outputs.shape):
                                    # 如果标签缺少通道维度，则添加
                                    if len(labels.shape) == 4 and len(outputs.shape) == 5:
                                        labels = labels.unsqueeze(1)  # 添加通道维度
                                    elif len(labels.shape) == 5 and len(outputs.shape) == 4:
                                        outputs = outputs.unsqueeze(1)  # 添加通道维度
                                else:
                                    # 如果维度数量相同但尺寸不同，尝试调整标签以匹配输出
                                    # 这里我们假设输出的尺寸是模型期望的尺寸
                                    # 我们需要将标签调整为与输出相同的尺寸
                                    # 使用最近邻插值方法调整标签尺寸，确保结果仍然是二值的
                                    if labels.shape[2:] != outputs.shape[2:]:
                                        # 只调整空间维度
                                        # 保存原始标签的数据类型
                                        original_dtype = labels.dtype
                                        # 转换为浮点型进行插值，然后转换回原始类型
                                        labels = F.interpolate(labels.float(), size=outputs.shape[2:], mode='nearest').to(original_dtype)
                            
                            loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        
                        pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
                        
            avg_val_loss = val_loss / len(val_loader)
            
            # 更新学习率调度器
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}: 训练损失={avg_train_loss:.4f}, 验证损失={avg_val_loss:.4f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_best_model(model, fold_idx, epoch, best_val_loss)
                patience_counter = 0  # 重置早停计数器
            else:
                patience_counter += 1
                
            # 早停检查
            if patience_counter >= max_patience:
                print(f"早停触发！第{fold_idx+1}折训练完成。")
                break
                
        # 记录结果
        result = {
            'fold': fold_idx + 1,
            'best_val_loss': best_val_loss,
            'train_indices': train_indices,
            'val_indices': val_indices
        }
        self.fold_results.append(result)
        
        return result
        
    def save_best_model(self, model, fold_idx, epoch, loss):
        """保存最佳模型
        
        保存当前折的最佳模型检查点，包含模型权重和训练元数据。
        
        参数:
            model (UNet3D): 模型实例
            fold_idx (int): 折索引 (0-indexed)
            epoch (int): 训练轮数
            loss (float): 损失值
            
        保存内容:
        - 模型权重 (model_state_dict)
        - 训练轮数 (epoch)
        - 折索引 (fold_idx)
        - 损失值 (loss)
        - 配置参数 (config)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'fold_idx': fold_idx,
            'loss': loss,
            'config': self.config
        }
        
        best_path = os.path.join(
            self.config['save_dir'],
            f'best_model_fold_{fold_idx}.pth'
        )
        torch.save(checkpoint, best_path)
        print(f"第{fold_idx+1}折最佳模型已保存至: {best_path}")
        
    def train(self):
        """执行完整的交叉验证训练
        
        对每一折进行训练和验证，最后汇总结果并保存统计信息。
        
        训练流程:
        1. 打印训练配置信息
        2. 遍历所有交叉验证折
        3. 对每折进行独立训练
        4. 保存交叉验证结果
        5. 打印训练总结
        """
        print(f"开始{self.n_splits}折交叉验证训练...")
        print(f"数据类型: {self.config.get('data_type', 'BPH')}")
        print(f"训练轮数: {self.config['num_epochs']}")
        print(f"批次大小: {self.config['batch_size']}")
        print(f"学习率: {self.config['learning_rate']}")
        
        # 训练每一折
        for fold_idx, (train_indices, val_indices) in enumerate(self.kfold_splits):
            self.train_fold(fold_idx, train_indices, val_indices)
            
        # 保存交叉验证结果
        self.save_results()
        
        # 打印总结
        self.print_summary()
        
    def save_results(self):
        """保存交叉验证结果
        
        将交叉验证的详细结果保存到JSON文件中，
        包含每折的最佳验证损失和统计信息。
        """
        results_path = os.path.join(self.config['save_dir'], 'cv_results.json')
        
        results = {
            'config': self.config,
            'fold_results': self.fold_results,
            'summary': {
                'mean_val_loss': sum(r['best_val_loss'] for r in self.fold_results) / len(self.fold_results),
                'std_val_loss': torch.tensor([r['best_val_loss'] for r in self.fold_results]).std().item(),
                'total_folds': len(self.fold_results)
            }
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        print(f"交叉验证结果已保存至: {results_path}")
        
    def print_summary(self):
        """打印交叉验证总结
        
        显示每折的最佳验证损失和总体统计信息。
        """
        print(f"\n{'='*60}")
        print("交叉验证训练完成！")
        print(f"{'='*60}")
        
        for result in self.fold_results:
            print(f"第{result['fold']}折: 最佳验证损失 = {result['best_val_loss']:.4f}")
            
        mean_loss = sum(r['best_val_loss'] for r in self.fold_results) / len(self.fold_results)
        print(f"\n平均验证损失: {mean_loss:.4f}")
        print(f"模型保存目录: {self.config['save_dir']}")
        print(f"{'='*60}")


def main():
    """主函数 - 配置和启动训练流程
    
    提供交互式训练模式选择，用户可以根据需求选择基础训练或交叉验证训练。
    
    功能:
    - 设置训练配置参数
    - 提供训练模式选择
    - 创建相应的训练器实例
    - 启动训练流程
    
    使用说明:
    1. 运行脚本
    2. 根据提示选择训练模式
    3. 等待训练完成
    4. 查看保存的模型和结果
    """
    # 基础配置
    config = {
        'data_dir': 'data',
        'num_epochs': 10,
        'batch_size': 4,  # 减少批次大小以解决内存不足问题
        'learning_rate': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data_type': 'BPH',  # 指定使用BPH数据进行训练
        'save_dir': os.path.join('checkpoints', f"BPH_CV_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        'n_splits': 5,  # 5折交叉验证
        'handle_missing_modalities': 'zero_fill',
        'validation': True
    }
    
    # 选择训练器类型
    print("选择训练模式:")
    print("1. 基础训练 (BPHTrainer)")
    print("2. 交叉验证训练 (CrossValidationTrainer)")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == '1':
        trainer = BPHTrainer(config)
    elif choice == '2':
        trainer = CrossValidationTrainer(config)
    else:
        print("无效选择，使用默认的基础训练器")
        trainer = BPHTrainer(config)
        
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()