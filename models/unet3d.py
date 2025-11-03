import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """双重3D卷积块
    
    包含两个连续的3D卷积层，每个卷积后接BatchNorm和ReLU激活函数
    用于U-Net中的特征提取
    """
    def __init__(self, in_channels, out_channels):
        """
        初始化双重3D卷积块
        
        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
        """
        super(DoubleConv3D, self).__init__()
        # 创建包含两个3D卷积层的序列模块
        # 每个卷积层后接批归一化和ReLU激活函数
        self.conv = nn.Sequential(
            # 第一个3D卷积层：3x3x3卷积核，padding=1保持尺寸不变
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            # 批归一化层，加速训练并提高稳定性
            nn.BatchNorm3d(out_channels),
            # ReLU激活函数，inplace=True节省内存
            nn.ReLU(inplace=True),
            # 第二个3D卷积层：同样使用3x3x3卷积核
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            # 批归一化层
            nn.BatchNorm3d(out_channels),
            # ReLU激活函数
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入张量，形状为(N, C, D, H, W)
            
        返回:
            torch.Tensor: 输出张量，形状为(N, out_channels, D, H, W)
        """
        return self.conv(x)

class Down3D(nn.Module):
    """下采样模块
    
    使用最大池化进行下采样，然后执行双重卷积
    在U-Net的编码器路径中使用
    """
    def __init__(self, in_channels, out_channels):
        """
        初始化下采样模块
        
        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
        """
        super(Down3D, self).__init__()
        # 创建包含最大池化和双重卷积的序列模块
        self.maxpool_conv = nn.Sequential(
            # 2x2x2最大池化层，将空间尺寸减半
            nn.MaxPool3d(2),
            # 双重3D卷积块
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入张量，形状为(N, C, D, H, W)
            
        返回:
            torch.Tensor: 输出张量，空间尺寸减半，通道数增加
        """
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """上采样模块
    
    使用转置卷积进行上采样，将特征图与编码器对应层的特征图拼接，然后执行双重卷积
    在U-Net的解码器路径中使用
    """
    def __init__(self, in_channels, out_channels):
        """
        初始化上采样模块
        
        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
        """
        super(Up3D, self).__init__()
        # 使用转置卷积进行上采样，将空间尺寸加倍
        # kernel_size=2, stride=2 实现2倍上采样
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # 双重3D卷积块处理拼接后的特征
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        前向传播函数
        
        参数:
            x1 (torch.Tensor): 来自上一层的特征图（需要上采样）
            x2 (torch.Tensor): 来自编码器对应层的特征图（需要拼接）
            
        返回:
            torch.Tensor: 上采样并融合后的特征图
        """
        # 对x1进行上采样
        x1 = self.up(x1)
        
        # 处理输入特征图尺寸不匹配的情况
        # 由于上采样和下采样过程中可能出现尺寸不匹配，需要进行填充对齐
        diff_z = x2.size()[2] - x1.size()[2]  # Z轴差异
        diff_y = x2.size()[3] - x1.size()[3]  # Y轴差异
        diff_x = x2.size()[4] - x1.size()[4]  # X轴差异
        
        # 对x1进行填充以匹配x2的尺寸
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2,
                       diff_z // 2, diff_z - diff_z // 2])
        
        # 在通道维度上拼接特征图
        # x2来自编码器，x1来自上采样，拼接后通道数翻倍
        x = torch.cat([x2, x1], dim=1)
        # 通过双重卷积处理拼接后的特征
        return self.conv(x)

class UNet3D(nn.Module):
    """3D U-Net模型
    
    用于前列腺多模态MRI图像分割的3D U-Net实现
    支持多模态输入(ADC、DWI、T2等)
    针对BPH数据进行了优化
    
    网络结构：
    - 编码器路径：包含4个下采样层，逐步提取特征并降低空间分辨率
    - 解码器路径：包含4个上采样层，逐步恢复空间分辨率并融合特征
    - 跳跃连接：将编码器各层特征传递给对应的解码器层
    """
    def __init__(self, n_modalities=5, n_classes=2):
        """
        初始化3D U-Net模型
        
        参数:
            n_modalities (int): 输入模态数量 (默认5: ADC、DWI、T2 fs、T2 not fs、gaoqing-T2)
            n_classes (int): 分割类别数 (默认2: 背景和前列腺病变)
        """
        super(UNet3D, self).__init__()
        self.n_modalities = n_modalities
        self.n_classes = n_classes
        
        # 初始特征通道数
        self.init_features = 64
        
        # 编码器部分（下采样路径）
        # 每一层将空间分辨率减半，特征通道数翻倍
        self.inc = DoubleConv3D(n_modalities, self.init_features)      # 输入层
        self.down1 = Down3D(self.init_features, self.init_features * 2)   # 第1个下采样层
        self.down2 = Down3D(self.init_features * 2, self.init_features * 4) # 第2个下采样层
        self.down3 = Down3D(self.init_features * 4, self.init_features * 8) # 第3个下采样层
        self.down4 = Down3D(self.init_features * 8, self.init_features * 16) # 第4个下采样层（最深层）
        
        # 解码器部分（上采样路径）
        # 每一层将空间分辨率加倍，特征通道数减半
        self.up1 = Up3D(self.init_features * 16, self.init_features * 8)  # 第1个上采样层
        self.up2 = Up3D(self.init_features * 8, self.init_features * 4)   # 第2个上采样层
        self.up3 = Up3D(self.init_features * 4, self.init_features * 2)   # 第3个上采样层
        self.up4 = Up3D(self.init_features * 2, self.init_features)       # 第4个上采样层
        
        # 输出层：1x1x1卷积将特征映射到类别空间
        self.outc = nn.Conv3d(self.init_features, n_classes, kernel_size=1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重
        
        使用Kaiming初始化卷积层权重，常数初始化批归一化层权重
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Kaiming初始化，适用于ReLU激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 如果有偏置项，则初始化为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                # 批归一化层权重初始化为1，偏置初始化为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """前向传播
        
        参数:
            x: 输入张量, shape为(batch_size, n_modalities, D, H, W)
        返回:
            分割预测图, shape为(batch_size, n_classes, D, H, W)
        """
        # 编码路径（下采样）
        # 每一层提取特征并降低分辨率
        x1 = self.inc(x)          # 初始特征提取 [N, 5, D, H, W] -> [N, 64, D, H, W]
        x2 = self.down1(x1)       # 第1次下采样 [N, 64, D, H, W] -> [N, 128, D/2, H/2, W/2]
        x3 = self.down2(x2)       # 第2次下采样 [N, 128, D/2, H/2, W/2] -> [N, 256, D/4, H/4, W/4]
        x4 = self.down3(x3)       # 第3次下采样 [N, 256, D/4, H/4, W/4] -> [N, 512, D/8, H/8, W/8]
        x5 = self.down4(x4)       # 第4次下采样 [N, 512, D/8, H/8, W/8] -> [N, 1024, D/16, H/16, W/16]
        
        # 解码路径（上采样）
        # 每一层恢复分辨率并融合来自编码器的特征
        x = self.up1(x5, x4)      # 第1次上采样 [N, 1024, D/16, H/16, W/16] -> [N, 512, D/8, H/8, W/8]
        x = self.up2(x, x3)       # 第2次上采样 [N, 512, D/8, H/8, W/8] -> [N, 256, D/4, H/4, W/4]
        x = self.up3(x, x2)       # 第3次上采样 [N, 256, D/4, H/4, W/4] -> [N, 128, D/2, H/2, W/2]
        x = self.up4(x, x1)       # 第4次上采样 [N, 128, D/2, H/2, W/2] -> [N, 64, D, H, W]
        
        # 生成分割预测
        # 通过1x1卷积将特征映射到类别空间
        logits = self.outc(x)     # 输出层 [N, 64, D, H, W] -> [N, n_classes, D, H, W]
        return logits

    def predict(self, x):
        """生成分割预测
        
        参数:
            x: 输入张量, shape为(batch_size, n_modalities, D, H, W)
        返回:
            分割预测的概率图, shape为(batch_size, n_classes, D, H, W)
        """
        # 设置为评估模式
        self.eval()
        # 关闭梯度计算以节省内存和计算资源
        with torch.no_grad():
            # 获取模型输出（logits）
            logits = self(x)
            # 使用sigmoid函数将logits转换为概率值（0-1之间）
            return torch.sigmoid(logits)

    def inference(self, x, threshold=0.5):
        """推理方法，直接输出二值化结果
        
        参数:
            x: 输入张量, shape为(batch_size, n_modalities, D, H, W)
            threshold: 阈值，大于此值的像素被分类为前景
        返回:
            二值化分割结果, shape为(batch_size, n_classes, D, H, W)
        """
        # 设置为评估模式
        self.eval()
        # 关闭梯度计算
        with torch.no_grad():
            # 获取模型输出（logits）
            logits = self(x)
            # 将logits转换为概率
            probs = torch.sigmoid(logits)
            # 根据阈值进行二值化处理
            return (probs > threshold).float()