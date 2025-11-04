"""
损失函数模块 - 提供医学图像分割任务专用的损失函数

该模块包含DiceLoss和BCEDiceLoss类，专门针对前列腺MRI分割任务设计，
能够有效处理医学图像分割中的类别不平衡问题。

作者: [项目作者]
版本: 1.0
创建时间: [项目创建时间]
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice损失函数
    
    专门用于医学图像分割任务的损失函数，基于Dice系数计算。
    Dice系数是医学图像分割中最常用的评估指标之一，
    能够有效处理前景和背景类别不平衡的问题。
    
    数学公式:
        Dice = (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
        Loss = 1 - Dice
    
    特性:
    - 对类别不平衡不敏感
    - 直接优化分割边界
    - 适用于二分类和多分类任务
    """
    def __init__(self, smooth=1.0):
        """
        初始化Dice损失函数
        
        参数:
            smooth (float): 平滑因子，防止除零错误，默认为1.0
                - 当预测值和真实值都为0时，避免分母为0
                - 值越大，对边界预测越平滑
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        """
        前向传播计算Dice损失
        
        参数:
            pred (torch.Tensor): 模型预测值，形状为(N, C, D, H, W)
                - N: 批次大小
                - C: 通道数（分割类别数）
                - D, H, W: 深度、高度、宽度
            target (torch.Tensor): 真实标签，形状为(N, C, D, H, W)
                - 与预测值形状相同，值为0或1
                
        返回:
            torch.Tensor: Dice损失值，标量
            
        计算步骤:
        1. 对预测值应用sigmoid激活，映射到[0,1]概率区间
        2. 展平张量便于计算交集和并集
        3. 计算预测值和真实值的交集
        4. 计算Dice系数
        5. 返回1 - Dice系数作为损失
        """
        # 检查预测值和目标值的形状是否匹配
        if pred.shape != target.shape:
            raise ValueError(f"预测值和目标值的形状不匹配: pred.shape={pred.shape}, target.shape={target.shape}")
        
        # 对预测值应用sigmoid激活函数，将值映射到[0,1]区间
        # 确保预测值表示概率，便于与二值标签比较
        pred = torch.sigmoid(pred)
        
        # 展平预测值和目标值，便于计算
        # 将5D张量展平为1D向量，便于计算交集和并集
        pred = pred.view(-1)
        target = target.view(-1)
        
        # 计算交集（预测值和真实值的元素级乘积之和）
        # 交集表示正确预测的正样本数量
        intersection = (pred * target).sum()
        
        # 计算Dice系数
        # 分子: 2倍交集 + 平滑因子
        # 分母: 预测值总和 + 真实值总和 + 平滑因子
        dice = (2. * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        
        # 返回Dice损失（1 - Dice系数）
        # Dice系数越大表示分割效果越好，因此损失越小
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """BCE和Dice的组合损失函数
    
    结合二元交叉熵损失和Dice损失的优点，提供更稳定的训练效果。
    BCE损失关注像素级别的分类准确性，
    Dice损失关注分割区域的整体相似度。
    
    组合优势:
    - BCE: 提供稳定的梯度，避免训练初期不稳定
    - Dice: 优化分割边界，处理类别不平衡
    - 综合两者优点，提高训练效率和分割质量
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        """
        初始化组合损失函数
        
        参数:
            bce_weight (float): BCE损失的权重，范围[0,1]
                - 控制BCE损失在总损失中的贡献
            dice_weight (float): Dice损失的权重，范围[0,1]
                - 控制Dice损失在总损失中的贡献
                - 通常设置bce_weight + dice_weight = 1
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        
    def forward(self, pred, target):
        """
        前向传播计算组合损失
        
        参数:
            pred (torch.Tensor): 模型预测值（logits）
                - 未经激活函数的原始输出
            target (torch.Tensor): 真实标签
                - 二值分割标签，值为0或1
                
        返回:
            torch.Tensor: 加权组合损失值
            
        计算流程:
        1. 分别计算BCE损失和Dice损失
        2. 按权重组合两个损失
        3. 返回加权和
        """
        # 计算二元交叉熵损失
        # BCEWithLogitsLoss内部包含sigmoid激活，适合处理logits
        bce_loss = self.bce_loss(pred, target)
        
        # 计算Dice损失
        # DiceLoss内部也包含sigmoid激活，确保一致性
        dice_loss = self.dice_loss(pred, target)
        
        # 返回加权组合损失
        # 权重可根据任务特点调整，通常各占0.5
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss