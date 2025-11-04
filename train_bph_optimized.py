import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.unet3d import UNet3D
from utils.losses import DiceLoss
from utils.trainer import BaseTrainer
from script.data_loader import get_dataloader
import os
from datetime import datetime
import json


class BPHTrainer(BaseTrainer):
    """BPH数据专用模型训练器
    
    继承自BaseTrainer，专门针对BPH数据进行优化
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
    
    使用K折交叉验证进行模型训练和评估
    """
    def __init__(self, config):
        """
        初始化交叉验证训练器
        
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
        self.device = torch.device(config['device'])
        self.n_splits = config.get('n_splits', 5)
        self.handle_missing = config.get('handle_missing_modalities', 'zero_fill')
        
        # 获取K折分割索引
        self.kfold_splits = get_kfold_splits(
            config['data_dir'],
            data_type=config.get('data_type', 'BPH'),
            n_splits=self.n_splits,
            handle_missing_modalities=self.handle_missing
        )
        
        # 记录每折的结果
        self.fold_results = []
        
        # 创建保存目录
        os.makedirs(config['save_dir'], exist_ok=True)
        
    def _create_model(self):
        """创建新的模型实例"""
        return UNet3D(
            n_modalities=5,  # 5个模态: ADC, DWI, T2 fs, T2 not fs, gaoqing-T2
            n_classes=1      # 二分类问题：背景和前列腺病变
        ).to(self.device)
        
    def _create_optimizer(self, model):
        """为模型创建优化器"""
        return optim.Adam(
            model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=1e-5  # 添加权重衰减以防止过拟合
        )
        
    def _create_scheduler(self, optimizer):
        """为优化器创建学习率调度器"""
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=10, 
            factor=0.5, 
            verbose=True
        )
        
    def train_fold(self, fold_idx, train_indices, val_indices):
        """训练单个折
        
        参数:
            fold_idx (int): 当前折的索引
            train_indices (list): 训练集索引列表
            val_indices (list): 验证集索引列表
            
        返回:
            dict: 该折的训练结果
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
            mode='train',
            data_type=self.config.get('data_type', 'BPH'),
            indices=train_indices,
            handle_missing_modalities=self.handle_missing
        )
        
        val_loader = get_dataloader(
            self.config['data_dir'],
            batch_size=self.config['batch_size'],
            mode='test',
            data_type=self.config.get('data_type', 'BPH'),
            indices=val_indices,
            handle_missing_modalities=self.handle_missing
        )
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15
        
        for epoch in range(self.config['num_epochs']):
            # 训练阶段
            model.train()
            train_loss = 0
            
            with tqdm(train_loader, desc=f'Fold {fold_idx+1} Training') as pbar:
                for batch in pbar:
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    # 前向传播
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    
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
                        
                        outputs = model(images)
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
                patience_counter = 0
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
        
        参数:
            model: 模型实例
            fold_idx (int): 折索引
            epoch (int): 训练轮数
            loss (float): 损失值
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
        """执行完整的交叉验证训练"""
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
        """保存交叉验证结果"""
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
        """打印交叉验证总结"""
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
    """主函数"""
    # 基础配置
    config = {
        'data_dir': 'data',
        'num_epochs': 100,
        'batch_size': 1,
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