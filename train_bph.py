import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.unet3d import UNet3D
from utils.losses import DiceLoss
from script.data_loader import get_dataloader
import numpy as np
from datetime import datetime

class BPHTrainer:
    """BPH数据专用模型训练器
    
    注意：对于小数据集（如240例），建议使用 train_bph_cv.py 中的交叉验证版本，
    以更好地利用数据并获得更稳定的模型性能评估。
    """
    def __init__(self, config):
        """
        初始化BPH训练器
        
        参数:
            config: 训练配置字典
                - data_dir: 数据集根目录
                - num_epochs: 训练轮数
                - batch_size: 批次大小
                - learning_rate: 学习率
                - device: 训练设备 ('cuda' or 'cpu')
                - save_dir: 模型保存目录
                - data_type: 数据类型 ('BPH' 或 'PCA')
                - handle_missing_modalities: 处理缺失模态的方法
        """
        self.config = config
        # 设置训练设备（GPU或CPU）
        self.device = torch.device(config['device'])
        
        # 创建3D U-Net模型
        # 输入5个模态，输出1个类别（背景和前列腺病变的二分类）
        self.model = UNet3D(
            n_modalities=5,  # 5个模态: ADC, DWI, T2 fs, T2 not fs, gaoqing-T2
            n_classes=1      # 二分类问题：背景和前列腺病变
        ).to(self.device)
        
        # 创建损失函数和优化器
        self.criterion = DiceLoss()
        # 使用Adam优化器，添加权重衰减以防止过拟合
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=1e-5  # 添加权重衰减以防止过拟合
        )
        
        # 学习率调度器
        # 当验证损失停止改善时，降低学习率
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',      # 当监控值最小化时
            patience=10,     # 等待10个epoch无改善后调整学习率
            factor=0.5,      # 学习率衰减因子
            verbose=True     # 打印学习率调整信息
        )
        
        # 获取处理缺失模态的策略
        handle_missing = config.get('handle_missing_modalities', 'zero_fill')
        
        # 创建训练数据加载器
        self.train_loader = get_dataloader(
            config['data_dir'],
            batch_size=config['batch_size'],
            mode='train',
            data_type=config.get('data_type', 'BPH'),  # 默认使用BPH数据
            handle_missing_modalities=handle_missing
        )
        
        # 创建验证数据加载器（如果有验证集）
        self.val_loader = None
        if config.get('validation', False):
            self.val_loader = get_dataloader(
                config['data_dir'],
                batch_size=config['batch_size'],
                mode='test',  # 假设测试模式即为验证模式
                data_type=config.get('data_type', 'BPH'),
                handle_missing_modalities=handle_missing
            )
        
        # 创建保存目录
        os.makedirs(config['save_dir'], exist_ok=True)
        
        # 训练历史记录
        self.train_history = {
            'loss': [],      # 训练损失历史
            'val_loss': []   # 验证损失历史
        }
        
    def train_epoch(self):
        """训练一个epoch
        
        返回:
            float: 该epoch的平均训练损失
        """
        # 设置模型为训练模式
        self.model.train()
        epoch_loss = 0
        
        # 使用tqdm显示训练进度
        with tqdm(self.train_loader, desc='Training') as pbar:
            for batch in pbar:
                # 将数据移动到指定设备（GPU或CPU）
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()  # 清零梯度
                outputs = self.model(images)  # 模型预测
                
                # 计算损失
                loss = self.criterion(outputs, labels)
                epoch_loss += loss.item()
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 更新模型参数
                self.optimizer.step()
                
                # 更新进度条显示
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
        # 返回平均损失
        return epoch_loss / len(self.train_loader)
    
    def validate_epoch(self):
        """验证一个epoch
        
        返回:
            float or None: 该epoch的平均验证损失，如果没有验证集则返回None
        """
        # 如果没有验证集，直接返回None
        if self.val_loader is None:
            return None
            
        # 设置模型为评估模式
        self.model.eval()
        val_loss = 0
        
        # 关闭梯度计算以节省内存和计算资源
        with torch.no_grad():
            # 使用tqdm显示验证进度
            for batch in tqdm(self.val_loader, desc='Validation'):
                # 将数据移动到指定设备
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 模型预测
                outputs = self.model(images)
                # 计算损失
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
        # 返回平均验证损失
        return val_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """保存检查点
        
        参数:
            epoch (int): 当前epoch数
            loss (float): 当前损失值
            is_best (bool): 是否为最佳模型
        """
        # 构建检查点字典
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
        }
        
        # 保存最新模型检查点
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
            # 只保存模型状态字典
            torch.save(self.model.state_dict(), best_path)
            print(f"最佳模型已保存至: {best_path}")
    
    def train(self):
        """完整的训练流程"""
        print(f"开始训练 {self.config.get('data_type', 'BPH')} 数据...")
        handle_method = self.config.get('handle_missing_modalities', 'zero_fill')
        print(f"处理缺失模态的方法: {handle_method}")
        print("注意：对于小数据集（如240例），建议使用 train_bph_cv.py 中的交叉验证版本，")
        print("以更好地利用数据并获得更稳定的模型性能评估。")
        # 初始化最佳损失为无穷大
        best_loss = float('inf')
        
        # 开始训练循环
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # 训练一个epoch
            train_loss = self.train_epoch()
            self.train_history['loss'].append(train_loss)
            
            # 验证（如果有验证集）
            val_loss = None
            if self.val_loader is not None:
                val_loss = self.validate_epoch()
                self.train_history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1} 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
                
                # 更新学习率（基于验证损失）
                self.scheduler.step(val_loss)
            else:
                print(f"Epoch {epoch+1} 训练损失: {train_loss:.4f}")
                # 更新学习率（基于训练损失）
                self.scheduler.step(train_loss)
            
            # 保存检查点
            is_best = train_loss < best_loss
            if is_best:
                best_loss = train_loss
                
            self.save_checkpoint(epoch + 1, train_loss, is_best)
            
        print(f"\n训练完成! 最佳损失: {best_loss:.4f}")

def main():
    """主函数"""
    # 训练配置参数
    config = {
        'data_dir': 'data',                    # 数据集目录
        'num_epochs': 100,                     # 训练轮数
        'batch_size': 1,                       # 批次大小
        'learning_rate': 1e-4,                 # 学习率
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 设备
        'data_type': 'BPH',                    # 数据类型
        'validation': False,                   # 是否使用验证集
        'handle_missing_modalities': 'zero_fill',  # 处理缺失模态的方法
        'save_dir': os.path.join('checkpoints', f"BPH_{datetime.now().strftime('%Y%m%d_%H%M%S')}")  # 模型保存目录
    }
    
    # 创建训练器并开始训练
    trainer = BPHTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()