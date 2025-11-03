import os
import torch
import numpy as np
from tqdm import tqdm
from models.unet3d import UNet3D
from script.data_loader import get_dataloader
import SimpleITK as sitk
from datetime import datetime
import json

def calculate_dice_score(pred, target):
    """计算Dice系数
    
    Dice系数用于衡量两个样本的相似性，值越接近1表示越相似
    
    参数:
        pred (torch.Tensor): 预测结果
        target (torch.Tensor): 真实标签
        
    返回:
        float: Dice系数
    """
    # 展平张量
    pred = pred.view(-1)
    target = target.view(-1)
    
    # 计算交集
    intersection = (pred * target).sum()
    # 计算Dice系数
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-8)
    return dice.item()

def calculate_iou(pred, target):
    """计算IoU (Intersection over Union)
    
    IoU也称为Jaccard指数，用于衡量两个样本的重叠程度
    
    参数:
        pred (torch.Tensor): 预测结果
        target (torch.Tensor): 真实标签
        
    返回:
        float: IoU值
    """
    # 展平张量
    pred = pred.view(-1)
    target = target.view(-1)
    
    # 计算交集
    intersection = (pred * target).sum()
    # 计算并集
    union = pred.sum() + target.sum() - intersection
    # 计算IoU
    iou = intersection / (union + 1e-8)
    return iou.item()

class ModelValidator:
    """模型验证器
    
    用于评估训练好的模型在测试集上的性能
    """
    
    def __init__(self, config):
        """
        初始化模型验证器
        
        参数:
            config: 配置字典
                - model_path: 模型文件路径
                - data_dir: 数据集目录
                - batch_size: 批次大小
                - device: 设备 ('cuda' or 'cpu')
                - data_type: 数据类型 ('BPH' 或 'PCA')
                - save_dir: 结果保存目录
                - handle_missing_modalities: 处理缺失模态的方法
        """
        self.config = config
        # 设置设备
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
            UNet3D: 加载的模型实例
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
            # 如果检查点包含模型状态字典
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果检查点直接是模型状态字典
            model.load_state_dict(checkpoint)
            
        # 设置为评估模式
        model.eval()
        print("模型加载完成!")
        return model
        
    def validate(self):
        """验证模型性能
        
        在测试集上评估模型性能，计算Dice系数和IoU等指标
        
        返回:
            tuple: (平均Dice系数, 平均IoU)
        """
        print("开始模型验证...")
        
        # 初始化统计变量
        total_dice = 0
        total_iou = 0
        count = 0
        
        # 存储每个病例的结果
        results = []
        
        # 关闭梯度计算
        with torch.no_grad():
            # 遍历测试数据
            for batch in tqdm(self.test_loader, desc='验证模型'):
                # 将数据移动到指定设备
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                case_ids = batch['case_id']
                
                # 模型预测
                outputs = self.model.predict(images)
                
                # 计算评估指标
                for i in range(outputs.shape[0]):
                    # 二值化预测结果
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
    """主函数"""
    # 验证配置参数
    config = {
        'model_path': '',  # 需要指定模型文件路径
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