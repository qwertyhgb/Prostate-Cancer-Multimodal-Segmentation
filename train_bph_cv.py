import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.unet3d import UNet3D
from utils.losses import DiceLoss
from script.data_loader import get_dataloader, get_kfold_splits
import numpy as np
from datetime import datetime
import json



class BPHCVTrainer:
    """BPH数据专用模型训练器（交叉验证版本）
    
    使用K折交叉验证训练模型，以更好地利用有限的数据并获得更稳定的模型性能评估。
    对于240例的小数据集，推荐使用5折交叉验证。
    """
    def __init__(self, config):
        """
        初始化BPH交叉验证训练器
        
        参数:
            config: 训练配置字典
                - data_dir: 数据集根目录
                - num_epochs: 训练轮数
                - batch_size: 批次大小
                - learning_rate: 学习率
                - device: 训练设备 ('cuda' or 'cpu')
                - save_dir: 模型保存目录
                - data_type: 数据类型 ('BPH' 或 'PCA')
                - n_splits: 交叉验证折数
                - handle_missing_modalities: 处理缺失模态的方法
        """
        self.config = config
        # 设置训练设备（GPU或CPU）
        self.device = torch.device(config['device'])
        # 获取交叉验证折数
        self.n_splits = config.get('n_splits', 5)
        # 获取处理缺失模态的策略
        self.handle_missing = config.get('handle_missing_modalities', 'zero_fill')
        
        # 获取交叉验证分割
        # 将数据集分割为K个不重叠的子集
        self.splits = get_kfold_splits(
            config['data_dir'],
            data_type=config.get('data_type', 'BPH'),
            n_splits=self.n_splits,
            handle_missing_modalities=self.handle_missing
        )
        
        # 记录每折的最好结果
        self.fold_results = []
        
        # 创建保存目录
        os.makedirs(config['save_dir'], exist_ok=True)
        
    def _create_model(self):
        """创建新的模型实例
        
        为每一折创建独立的模型实例，避免交叉验证中的数据泄露问题
        
        返回:
            UNet3D: 新的3D U-Net模型实例
        """
        model = UNet3D(
            n_modalities=5,  # 5个模态: ADC, DWI, T2 fs, T2 not fs, gaoqing-T2
            n_classes=1      # 二分类问题：背景和前列腺病变
        ).to(self.device)
        return model
        
    def _create_optimizer(self, model):
        """为模型创建优化器
        
        参数:
            model (UNet3D): 模型实例
            
        返回:
            Adam: Adam优化器实例
        """
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=1e-5  # 添加权重衰减以防止过拟合
        )
        return optimizer
        
    def _create_scheduler(self, optimizer):
        """为优化器创建学习率调度器
        
        参数:
            optimizer (Adam): 优化器实例
            
        返回:
            ReduceLROnPlateau: 学习率调度器实例
        """
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=10, 
            factor=0.5, 
            verbose=True
        )
        return scheduler
        
    def train_fold(self, fold_idx, train_indices, val_indices):
        """训练单个折
        
        参数:
            fold_idx (int): 当前折的索引
            train_indices (list): 训练集索引列表
            val_indices (list): 验证集索引列表
            
        返回:
            float: 该折的最佳验证损失
        """
        print(f"\n{'='*50}")
        print(f"开始训练第 {fold_idx + 1}/{self.n_splits} 折")
        print(f"训练集大小: {len(train_indices)}, 验证集大小: {len(val_indices)}")
        print(f"处理缺失模态的方法: {self.handle_missing}")
        print(f"{'='*50}")
        
        # 创建模型、优化器和调度器
        # 为每一折创建独立的模型实例
        model = self._create_model()
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer)
        criterion = DiceLoss()
        
        # 创建数据加载器
        # 训练数据加载器
        train_loader = get_dataloader(
            self.config['data_dir'],
            batch_size=self.config['batch_size'],
            mode='train',
            data_type=self.config.get('data_type', 'BPH'),
            fold_indices=train_indices,
            handle_missing_modalities=self.handle_missing
        )
        
        # 验证数据加载器
        val_loader = get_dataloader(
            self.config['data_dir'],
            batch_size=self.config['batch_size'],
            mode='test',
            data_type=self.config.get('data_type', 'BPH'),
            fold_indices=val_indices,
            handle_missing_modalities=self.handle_missing
        )
        
        # 初始化最佳验证损失
        best_val_loss = float('inf')
        # 训练历史记录
        train_history = {'train_loss': [], 'val_loss': []}
        
        # 开始训练循环
        for epoch in range(self.config['num_epochs']):
            # 训练一个epoch
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            train_history['train_loss'].append(train_loss)
            
            # 验证一个epoch
            val_loss = self.validate_epoch(model, val_loader, criterion)
            train_history['val_loss'].append(val_loss)
            
            # 打印训练信息
            print(f"Fold {fold_idx+1} Epoch {epoch+1}/{self.config['num_epochs']} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, val_loss, 
                    fold_idx, is_best=True
                )
        
        # 保存训练历史
        history_path = os.path.join(
            self.config['save_dir'], 
            f'fold_{fold_idx}_history.json'
        )
        with open(history_path, 'w') as f:
            json.dump(train_history, f)
            
        return best_val_loss
        
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """训练一个epoch
        
        参数:
            model (UNet3D): 模型实例
            train_loader (DataLoader): 训练数据加载器
            optimizer (Adam): 优化器实例
            criterion (DiceLoss): 损失函数
            
        返回:
            float: 该epoch的平均训练损失
        """
        # 设置模型为训练模式
        model.train()
        epoch_loss = 0
        
        # 使用tqdm显示训练进度
        with tqdm(train_loader, desc='Training', leave=False) as pbar:
            for batch in pbar:
                # 将数据移动到指定设备
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = model(images)
                
                # 计算损失
                loss = criterion(outputs, labels)
                epoch_loss += loss.item()
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 更新模型参数
                optimizer.step()
                
                # 更新进度条显示
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
        # 返回平均损失
        return epoch_loss / len(train_loader)
    
    def validate_epoch(self, model, val_loader, criterion):
        """验证一个epoch
        
        参数:
            model (UNet3D): 模型实例
            val_loader (DataLoader): 验证数据加载器
            criterion (DiceLoss): 损失函数
            
        返回:
            float: 该epoch的平均验证损失
        """
        # 设置模型为评估模式
        model.eval()
        val_loss = 0
        
        # 关闭梯度计算以节省内存和计算资源
        with torch.no_grad():
            # 使用tqdm显示验证进度
            for batch in tqdm(val_loader, desc='Validation', leave=False):
                # 将数据移动到指定设备
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 模型预测
                outputs = model(images)
                # 计算损失
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        # 返回平均验证损失
        return val_loss / len(val_loader)
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss, fold_idx, is_best=False):
        """保存检查点
        
        参数:
            model (UNet3D): 模型实例
            optimizer (Adam): 优化器实例
            scheduler (ReduceLROnPlateau): 学习率调度器实例
            epoch (int): 当前epoch数
            loss (float): 当前损失值
            fold_idx (int): 当前折索引
            is_best (bool): 是否为最佳模型
        """
        # 只保存最佳模型
        if is_best:
            # 构建检查点字典
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
                'fold_idx': fold_idx
            }
            
            # 保存最佳模型
            best_path = os.path.join(
                self.config['save_dir'],
                f'best_model_fold_{fold_idx}.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"第 {fold_idx + 1} 折最佳模型已保存至: {best_path}")
    
    def train(self):
        """完整的交叉验证训练流程
        
        对每一折进行训练和验证，最后汇总结果
        """
        print(f"开始 {self.n_splits} 折交叉验证训练 {self.config.get('data_type', 'BPH')} 数据...")
        print(f"处理缺失模态的方法: {self.handle_missing}")
        
        # 存储所有折的结果
        all_fold_results = []
        
        # 训练每一折
        for fold_idx, (train_indices, val_indices) in enumerate(self.splits):
            # 训练当前折并获取最佳验证损失
            best_val_loss = self.train_fold(fold_idx, train_indices, val_indices)
            # 记录结果
            all_fold_results.append({
                'fold': fold_idx,
                'best_val_loss': best_val_loss
            })
            
        # 保存总体结果
        results_path = os.path.join(self.config['save_dir'], 'cv_results.json')
        with open(results_path, 'w') as f:
            json.dump(all_fold_results, f)
            
        # 计算平均结果和标准差
        avg_loss = np.mean([r['best_val_loss'] for r in all_fold_results])
        std_loss = np.std([r['best_val_loss'] for r in all_fold_results])
        
        # 打印最终结果
        print(f"\n{'='*50}")
        print(f"交叉验证训练完成!")
        print(f"各折最佳验证损失:")
        for r in all_fold_results:
            print(f"  折 {r['fold'] + 1}: {r['best_val_loss']:.4f}")
        print(f"平均损失: {avg_loss:.4f} ± {std_loss:.4f}")
        print(f"{'='*50}")
        
        return avg_loss, std_loss

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
        'n_splits': 5,                         # 5折交叉验证
        'handle_missing_modalities': 'zero_fill',  # 处理缺失模态的方法
        'save_dir': os.path.join('checkpoints', f"BPH_CV_{datetime.now().strftime('%Y%m%d_%H%M%S')}")  # 模型保存目录
    }
    
    # 创建训练器并开始训练
    trainer = BPHCVTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()