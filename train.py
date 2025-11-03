import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.unet3d import UNet3D
from script.data_loader import get_dataloader
import numpy as np
from datetime import datetime

class DiceLoss(nn.Module):
    """Dice损失函数
    
    用于评估分割预测与真实标签之间的相似度
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # 展平预测值和目标值
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        return 1 - dice

class Trainer:
    """模型训练器"""
    def __init__(self, config):
        """
        参数:
            config: 训练配置字典
                - data_dir: 数据集目录
                - num_epochs: 训练轮数
                - batch_size: 批次大小
                - learning_rate: 学习率
                - device: 训练设备 ('cuda' or 'cpu')
                - save_dir: 模型保存目录
                - data_type: 数据类型 ('BPH' 或 'PCA')
        """
        self.config = config
        self.device = torch.device(config['device'])
        
        # 创建模型
        self.model = UNet3D(
            n_modalities=5,  # 5个模态: ADC, DWI, T2 fs, T2 not fs, gaoqing-T2
            n_classes=1      # 二分类问题：背景和前列腺病变
        ).to(self.device)
        
        # 创建损失函数和优化器
        self.criterion = DiceLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate']
        )
        
        # 创建数据加载器
        self.train_loader = get_dataloader(
            config['data_dir'],
            batch_size=config['batch_size'],
            mode='train',
            data_type=config.get('data_type', 'BPH')  # 默认使用BPH数据
        )
        
        # 创建保存目录
        os.makedirs(config['save_dir'], exist_ok=True)
        
    def train_epoch(self):
        """训练一个epoch"""
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
    
    def save_checkpoint(self, epoch, loss):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        
        checkpoint_path = os.path.join(
            self.config['save_dir'],
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
    def train(self):
        """完整的训练流程"""
        print(f"开始训练 {self.config.get('data_type', 'BPH')} 数据...")
        best_loss = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # 训练一个epoch
            loss = self.train_epoch()
            
            # 保存最佳模型
            if loss < best_loss:
                best_loss = loss
                self.save_checkpoint(epoch + 1, loss)
                print(f"保存最佳模型 (loss: {loss:.4f})")
            
            print(f"Epoch {epoch+1} 平均损失: {loss:.4f}")

def main():
    """主函数"""
    config = {
        'data_dir': 'data',
        'num_epochs': 100,
        'batch_size': 1,
        'learning_rate': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data_type': 'BPH',  # 指定使用BPH数据进行训练
        'save_dir': os.path.join('checkpoints', f"BPH_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    }
    
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()