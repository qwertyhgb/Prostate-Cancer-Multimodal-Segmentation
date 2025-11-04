"""
前列腺多模态MRI分割项目 - 配置文件示例

这个文件包含了所有可用的配置参数，用户可以根据自己的需求进行修改。
"""

import torch

# 基础训练配置
BASE_CONFIG = {
    # 数据相关
    'data_dir': 'data',                    # 数据目录路径
    'data_type': 'BPH',                    # 数据类型: 'BPH' 或 'PCA'
    'handle_missing_modalities': 'zero_fill',  # 缺失模态处理: 'zero_fill', 'skip', 'duplicate'
    
    # 训练参数
    'num_epochs': 100,                     # 训练轮数
    'batch_size': 1,                       # 批次大小 (根据GPU内存调整)
    'learning_rate': 1e-4,                 # 学习率
    'validation': True,                    # 是否使用验证集
    
    # 硬件配置
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 训练设备
    
    # 模型保存
    'save_dir': 'checkpoints',             # 模型保存目录
    'save_frequency': 10,                  # 保存频率 (每N个epoch保存一次)
    
    # 早停机制
    'early_stopping': True,                # 是否使用早停
    'patience': 15,                        # 早停耐心值
    
    # 日志和监控
    'log_frequency': 1,                    # 日志频率 (每N个batch记录一次)
    'print_frequency': 10,                 # 打印频率 (每N个batch打印一次)
}

# 交叉验证配置
CROSS_VALIDATION_CONFIG = {
    **BASE_CONFIG,  # 继承基础配置
    'n_splits': 5,                         # 交叉验证折数
    'stratified': True,                  # 是否使用分层采样
}

# 快速训练配置 (用于测试)
QUICK_TRAIN_CONFIG = {
    **BASE_CONFIG,  # 继承基础配置
    'num_epochs': 10,                      # 减少训练轮数
    'batch_size': 2,                       # 增加批次大小
    'validation': False,                   # 关闭验证以加快速度
    'early_stopping': False,               # 关闭早停
}

# 高性能训练配置 (用于正式训练)
HIGH_PERFORMANCE_CONFIG = {
    **BASE_CONFIG,  # 继承基础配置
    'num_epochs': 200,                     # 增加训练轮数
    'batch_size': 4,                       # 增加批次大小 (需要更多GPU内存)
    'learning_rate': 5e-5,                 # 降低学习率
    'patience': 20,                        # 增加早停耐心值
    'save_frequency': 5,                   # 增加保存频率
}

# 小数据集配置 (适用于数据量较少的情况)
SMALL_DATASET_CONFIG = {
    **CROSS_VALIDATION_CONFIG,  # 继承交叉验证配置
    'batch_size': 1,                       # 小批次大小
    'n_splits': 10,                        # 增加交叉验证折数
    'learning_rate': 1e-4,                 # 保持较高学习率
    'data_augmentation': True,             # 启用数据增强
}

# 模型架构配置
MODEL_CONFIG = {
    'n_modalities': 5,                     # 输入模态数量 (ADC, DWI, T2 fs, T2 not fs, gaoqing-T2)
    'n_classes': 1,                        # 输出类别数量 (二分类)
    'base_channels': 64,                   # 基础通道数
    'dropout_rate': 0.1,                   # dropout率
    'use_batch_norm': True,                # 是否使用批归一化
    'use_residual': False,                 # 是否使用残差连接
}

# 优化器配置
OPTIMIZER_CONFIG = {
    'optimizer': 'Adam',                   # 优化器类型: 'Adam', 'SGD', 'RMSprop'
    'weight_decay': 1e-5,                  # 权重衰减
    'momentum': 0.9,                     # 动量 (仅SGD使用)
    'betas': (0.9, 0.999),               # Adam的beta参数
    'eps': 1e-8,                         # Adam的epsilon参数
}

# 学习率调度器配置
SCHEDULER_CONFIG = {
    'scheduler': 'ReduceLROnPlateau',      # 调度器类型
    'mode': 'min',                       # 模式: 'min' 或 'max'
    'factor': 0.5,                       # 学习率衰减因子
    'patience': 10,                        # 耐心值
    'threshold': 0.0001,                 # 阈值
    'cooldown': 0,                       # 冷却时间
    'min_lr': 1e-7,                      # 最小学习率
}

# 损失函数配置
LOSS_CONFIG = {
    'loss_function': 'DiceLoss',           # 损失函数: 'DiceLoss', 'BCEDiceLoss', 'BCEWithLogitsLoss'
    'dice_smooth': 1e-5,                 # Dice损失的平滑因子
    'bce_weight': 0.5,                   # BCE损失的权重 (用于BCEDiceLoss)
    'dice_weight': 0.5,                  # Dice损失的权重 (用于BCEDiceLoss)
}

# 数据增强配置
AUGMENTATION_CONFIG = {
    'enabled': False,                      # 是否启用数据增强
    'rotation_range': 10,                  # 旋转角度范围 (度)
    'width_shift_range': 0.1,              # 宽度平移范围
    'height_shift_range': 0.1,             # 高度平移范围
    'zoom_range': 0.1,                   # 缩放范围
    'horizontal_flip': True,               # 是否水平翻转
    'vertical_flip': True,                 # 是否垂直翻转
    'fill_mode': 'nearest',                # 填充模式
}

# 完整的训练配置 (整合所有配置)
COMPLETE_CONFIG = {
    **BASE_CONFIG,
    **MODEL_CONFIG,
    **OPTIMIZER_CONFIG,
    **SCHEDULER_CONFIG,
    **LOSS_CONFIG,
    **AUGMENTATION_CONFIG,
}

# 预设配置组合
PRESET_CONFIGS = {
    'quick': QUICK_TRAIN_CONFIG,           # 快速测试
    'standard': BASE_CONFIG,               # 标准训练
    'cross_validation': CROSS_VALIDATION_CONFIG,  # 交叉验证
    'high_performance': HIGH_PERFORMANCE_CONFIG,  # 高性能训练
    'small_dataset': SMALL_DATASET_CONFIG,       # 小数据集
}

def get_config(preset='standard', **kwargs):
    """
    获取配置
    
    参数:
        preset (str): 预设配置名称 ('quick', 'standard', 'cross_validation', 'high_performance', 'small_dataset')
        **kwargs: 额外的配置参数，将覆盖预设配置
    
    返回:
        dict: 配置字典
    """
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"未知的预设配置: {preset}. 可用选项: {list(PRESET_CONFIGS.keys())}")
    
    config = PRESET_CONFIGS[preset].copy()
    config.update(kwargs)
    
    return config

# 使用示例
if __name__ == "__main__":
    # 获取标准配置
    config = get_config('standard')
    print("标准配置:", config)
    
    # 获取交叉验证配置并自定义参数
    cv_config = get_config('cross_validation', num_epochs=150, n_splits=10)
    print("交叉验证配置:", cv_config)
    
    # 获取快速训练配置
    quick_config = get_config('quick', data_type='PCA')
    print("快速训练配置:", quick_config)