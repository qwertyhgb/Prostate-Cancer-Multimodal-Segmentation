import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice损失函数
    
    用于评估分割预测与真实标签之间的相似度
    Dice系数是医学图像分割中常用的评估指标
    """
    def __init__(self, smooth=1.0):
        """
        初始化Dice损失函数
        
        参数:
            smooth (float): 平滑因子，防止除零错误，默认为1.0
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        """
        前向传播计算Dice损失
        
        参数:
            pred (torch.Tensor): 模型预测值，形状为(N, C, D, H, W)
            target (torch.Tensor): 真实标签，形状为(N, C, D, H, W)
            
        返回:
            torch.Tensor: Dice损失值
        """
        # 对预测值应用sigmoid激活函数，将值映射到[0,1]区间
        pred = torch.sigmoid(pred)
        
        # 展平预测值和目标值，便于计算
        pred = pred.view(-1)
        target = target.view(-1)
        
        # 计算交集（预测值和真实值的元素级乘积之和）
        intersection = (pred * target).sum()
        
        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        
        # 返回Dice损失（1 - Dice系数）
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """BCE和Dice的组合损失函数
    
    结合二元交叉熵损失和Dice损失，综合两者的优点
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        """
        初始化组合损失函数
        
        参数:
            bce_weight (float): BCE损失的权重
            dice_weight (float): Dice损失的权重
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
            pred (torch.Tensor): 模型预测值
            target (torch.Tensor): 真实标签
            
        返回:
            torch.Tensor: 组合损失值
        """
        bce_loss = self.bce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss