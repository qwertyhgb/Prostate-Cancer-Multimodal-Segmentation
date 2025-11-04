"""
训练器模块 - 提供模型训练、验证和保存功能

该模块包含BaseTrainer类，用于管理3D U-Net模型的完整训练流程，
包括数据加载、模型训练、验证、学习率调度、早停机制和模型保存。

作者: [项目作者]
版本: 1.0
创建时间: [项目创建时间]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.unet3d import UNet3D
from utils.losses import DiceLoss
from script.data_loader import get_dataloader
import os
from datetime import datetime


class BaseTrainer:
    """基础训练器类
    
    提供通用的训练功能，包括模型创建、优化器设置、训练循环等。
    支持早停机制、学习率调度、模型保存等高级训练特性。
    
    特性:
    - 完整的训练和验证流程管理
    - 自动学习率调度和早停机制
    - 模型检查点保存和恢复
    - 进度条显示和训练状态监控
    """
    def __init__(self, config):
        """
        初始化基础训练器
        
        参数:
            config (dict): 训练配置字典，包含以下键值:
                - data_dir (str): 数据集根目录路径
                - num_epochs (int): 训练总轮数
                - batch_size (int): 每个批次的样本数量
                - learning_rate (float): 优化器学习率
                - device (str): 训练设备 ('cuda' 或 'cpu')
                - save_dir (str): 模型和检查点保存目录
                - data_type (str): 数据类型 ('BPH' 或 'PCA')
                - handle_missing_modalities (str): 处理缺失模态的方法
                - validation (bool): 是否启用验证集验证
        
        初始化流程:
        1. 设置训练设备
        2. 创建模型实例
        3. 配置损失函数和优化器
        4. 创建数据加载器
        5. 创建保存目录
        """
        self.config = config
        self.device = torch.device(config['device'])
        
        # 创建模型
        self.model = self._create_model()
        
        # 创建损失函数和优化器
        self.criterion = self._create_criterion()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 创建数据加载器
        self.train_loader = self._create_dataloader('train')
        self.val_loader = self._create_dataloader('test') if config.get('validation', False) else None
        
        # 创建保存目录
        os.makedirs(config['save_dir'], exist_ok=True)
        
    def _create_model(self):
        """创建3D U-Net模型实例
        
        返回:
            UNet3D: 配置好的3D U-Net模型实例
            
        模型配置说明:
        - n_modalities=5: 支持5种MRI模态输入
        - n_classes=1: 二分类问题（背景和前列腺病变）
        """
        return UNet3D(
            n_modalities=5,  # 5个模态: ADC, DWI, T2 fs, T2 not fs, gaoqing-T2
            n_classes=1      # 二分类问题：背景和前列腺病变
        ).to(self.device)
        
    def _create_criterion(self):
        """创建损失函数
        
        返回:
            DiceLoss: Dice损失函数实例
            
        说明:
        Dice损失函数特别适合医学图像分割任务，
        能够有效处理类别不平衡问题。
        """
        return DiceLoss()
        
    def _create_optimizer(self):
        """创建优化器
        
        返回:
            optim.Adam: Adam优化器实例
            
        配置说明:
        - 使用Adam优化器，结合了动量方法和自适应学习率
        - weight_decay=1e-5: 添加L2正则化防止过拟合
        """
        return optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=1e-5  # 添加权重衰减以防止过拟合
        )
        
    def _create_scheduler(self):
        """创建学习率调度器
        
        返回:
            optim.lr_scheduler.ReduceLROnPlateau: 学习率调度器实例
            
        调度策略:
        - mode='min': 当监控的损失值最小化时触发
        - patience=10: 等待10个epoch无改善后调整学习率
        - factor=0.5: 每次调整将学习率减半
        - verbose=True: 打印学习率调整信息
        """
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',      # 当监控值最小化时
            patience=10,     # 等待10个epoch无改善后调整学习率
            factor=0.5,      # 学习率衰减因子
            verbose=True     # 打印学习率调整信息
        )
        
    def _create_dataloader(self, mode):
        """创建数据加载器
        
        参数:
            mode (str): 数据模式 ('train' 或 'test')
            
        返回:
            DataLoader: PyTorch数据加载器实例
            
        功能:
        根据配置创建训练或验证数据加载器，
        自动处理缺失模态和数据类型选择。
        """
        return get_dataloader(
            self.config['data_dir'],
            batch_size=self.config['batch_size'],
            mode=mode,
            data_type=self.config.get('data_type', 'BPH'),
            handle_missing_modalities=self.config.get('handle_missing_modalities', 'zero_fill')
        )
        
    def train_epoch(self):
        """训练一个epoch
        
        返回:
            float: 该epoch的平均训练损失
            
        训练流程:
        1. 设置模型为训练模式
        2. 遍历训练数据加载器
        3. 前向传播计算输出
        4. 计算损失并反向传播
        5. 更新模型参数
        6. 更新进度条显示
        """
        self.model.train()
        epoch_loss = 0
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(images)
                
                # 计算损失
                loss = self.criterion(outputs, labels)
                epoch_loss += loss.item()
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
        return epoch_loss / len(self.train_loader)
        
    def validate_epoch(self):
        """验证一个epoch
        
        返回:
            float or None: 该epoch的平均验证损失，如果没有验证集则返回None
            
        验证流程:
        1. 设置模型为评估模式
        2. 禁用梯度计算以节省内存
        3. 遍历验证数据加载器
        4. 计算验证损失
        5. 更新进度条显示
        """
        if self.val_loader is None:
            return None
            
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation') as pbar:
                for batch in pbar:
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    # 模型预测
                    outputs = self.model(images)
                    
                    # 计算损失
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    # 更新进度条
                    pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
                    
        return val_loss / len(self.val_loader)
        
    def save_checkpoint(self, epoch, loss, is_best=False):
        """保存检查点
        
        参数:
            epoch (int): 当前epoch数
            loss (float): 当前损失值
            is_best (bool): 是否为最佳模型
            
        保存内容:
        - 模型状态字典
        - 优化器状态字典
        - 调度器状态字典
        - 训练配置
        - 当前epoch和损失
        
        文件命名:
        - 最新检查点: latest_checkpoint.pth
        - 最佳模型: best_model_epoch_{epoch}.pth
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(
            self.config['save_dir'],
            'latest_checkpoint.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，则额外保存
        if is_best:
            best_path = os.path.join(
                self.config['save_dir'],
                f'best_model_epoch_{epoch}.pth'
            )
            torch.save(self.model.state_dict(), best_path)
            print(f"最佳模型已保存至: {best_path}")
            
    def train(self):
        """完整的训练流程
        
        训练流程:
        1. 打印训练配置信息
        2. 循环训练指定轮数
        3. 每个epoch进行训练和验证
        4. 更新学习率调度器
        5. 保存最佳模型检查点
        6. 实现早停机制
        
        早停机制:
        - 监控验证损失（如有）或训练损失
        - 连续20个epoch无改善则停止训练
        - 保存最佳模型状态
        """
        print(f"开始训练 {self.config.get('data_type', 'BPH')} 数据...")
        print(f"处理缺失模态的方法: {self.config.get('handle_missing_modalities', 'zero_fill')}")
        print(f"训练轮数: {self.config['num_epochs']}")
        print(f"批次大小: {self.config['batch_size']}")
        print(f"学习率: {self.config['learning_rate']}")
        print(f"设备: {self.config['device']}")
        print(f"保存目录: {self.config['save_dir']}")
        
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 20  # 早停耐心值
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # 训练一个epoch
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate_epoch()
            
            # 打印训练信息
            if val_loss is not None:
                print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
                # 更新学习率调度器
                self.scheduler.step(val_loss)
                # 检查是否为最佳模型
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_checkpoint(epoch + 1, val_loss, is_best=True)
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                print(f"训练损失: {train_loss:.4f}")
                self.scheduler.step(train_loss)
                if train_loss < best_loss:
                    best_loss = train_loss
                    self.save_checkpoint(epoch + 1, train_loss, is_best=True)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
            # 早停检查
            if patience_counter >= max_patience:
                print(f"\n早停触发！连续{max_patience}个epoch没有改善。")
                break
                
        print(f"\n训练完成！最佳损失: {best_loss:.4f}")
        print(f"模型保存在: {self.config['save_dir']}")