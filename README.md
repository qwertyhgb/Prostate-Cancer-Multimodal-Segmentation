# 前列腺多模态MRI分割项目

这是一个基于3D U-Net的前列腺多模态MRI图像分割项目，支持BPH和PCA数据的自动分割。

## 项目结构

```
U-Net_3D/
├── models/                  # 模型定义
│   └── unet3d.py           # 3D U-Net模型实现
├── script/                  # 数据处理和辅助脚本
│   ├── data_loader.py      # 数据加载器
│   ├── validate_model.py   # 模型验证脚本
│   └── predict.py          # 模型预测脚本
├── utils/                   # 工具模块
│   ├── losses.py           # 损失函数
│   └── trainer.py          # 基础训练器
├── train_bph.py            # BPH数据训练脚本
├── train_bph_cv.py         # BPH数据交叉验证训练脚本
├── train_bph_optimized.py  # 优化版训练脚本（推荐）
├── data/                   # 数据目录
│   └── BPH-PCA/           # BPH和PCA数据
│       ├── BPH/           # BPH数据
│       │   ├── ADC/       # ADC模态图像
│       │   ├── DWI/       # DWI模态图像
│       │   ├── T2 fs/     # T2脂肪抑制模态图像
│       │   ├── T2 not fs/ # T2非脂肪抑制模态图像
│       │   └── gaoqing-T2/ # 高清T2模态图像
│       ├── PCA/           # PCA数据
│       └── ROI(BPH+PCA)/ # 分割标签
└── checkpoints/            # 模型检查点
```

## 功能特性

- **多模态支持**：支持5种MRI模态（ADC、DWI、T2 fs、T2 not fs、gaoqing-T2）
- **数据类型支持**：支持BPH和PCA两种数据类型
- **缺失模态处理**：支持零填充、跳过等多种缺失模态处理策略
- **交叉验证**：提供K折交叉验证功能
- **早停机制**：防止过拟合
- **模型保存**：自动保存最佳模型和检查点

## 快速开始

### 1. 环境检查

首先检查您的运行环境是否满足要求：

```bash
python run.py check
```

### 2. 安装依赖

#### 方法一：使用requirements.txt（推荐）
```bash
pip install -r requirements.txt
```

#### 方法二：手动安装
```bash
pip install torch torchvision
pip install SimpleITK
pip install tqdm
pip install scikit-learn
pip install numpy
```

### 3. 项目检查（推荐）

安装依赖后，建议运行项目检查以确保一切正常：

```bash
python check_project.py
```

这个脚本会检查：
- 项目文件结构完整性
- 依赖包安装状态
- 模型文件（如果存在）
- 数据目录结构
- 生成详细的检查报告

### 4. 数据准备

确保数据目录结构如下：

```
data/
└── BPH-PCA/
    ├── BPH/
    │   ├── ADC/
    │   ├── DWI/
    │   ├── T2 fs/
    │   ├── T2 not fs/
    │   └── gaoqing-T2/
    ├── PCA/
    │   ├── ADC/
    │   ├── DWI/
    │   ├── T2 fs/
    │   ├── T2 not fs/
    │   └── gaoqing-T2/
    └── ROI(BPH+PCA)/
        ├── BPH/
        └── PCA/
```

#### 数据格式要求

- **图像格式**: NIfTI (.nii或.nii.gz) 或 MetaImage (.mha)
- **模态对齐**: 所有模态图像必须在相同的空间坐标系中
- **尺寸一致**: 所有模态图像的空间尺寸必须相同
- **标签格式**: 二值图像，前景为1，背景为0

#### 缺失模态处理

项目支持多种缺失模态处理策略：

1. **zero_fill** (默认): 用零填充缺失模态
2. **skip**: 跳过不完整的病例
3. **duplicate**: 使用参考模态填充

### 3. 训练模型

#### 使用统一运行脚本（推荐）
```bash
# 基础训练
python run.py train --config optimized --data_type BPH

# 交叉验证训练
python run.py train --cross_validation --config optimized --data_type BPH

# 快速训练（用于测试）
python run.py train --config optimized --data_type BPH --epochs 10

# 高性能训练
python run.py train --config optimized --data_type BPH --epochs 200 --batch_size 4
```

#### 使用独立训练脚本
```bash
# 使用优化版训练脚本（推荐）
python train_bph_optimized.py

# 使用基础训练脚本
python train_bph.py

# 使用交叉验证训练
python train_bph_cv.py
```

### 4. 验证模型

#### 使用统一运行脚本（推荐）
```bash
# 验证最佳模型
python run.py validate --model_path checkpoints/best_model.pth

# 验证指定模型和数据类型
python run.py validate --model_path checkpoints/best_model.pth --data_type BPH
```

#### 使用独立验证脚本
```bash
python script/validate_model.py
```

### 5. 预测新数据

#### 使用统一运行脚本（推荐）
```bash
# 预测单个病例
python run.py predict --model_path checkpoints/best_model.pth --input_data data/test_case --output_path results/

# 预测多个病例
python run.py predict --model_path checkpoints/best_model.pth --input_data data/test_cases/ --output_path results/
```

#### 使用独立预测脚本
```bash
python script/predict.py
```

## 训练配置

### 使用配置文件

项目提供了详细的配置文件示例 (`config_example.py`)，包含多种预设配置：

```python
# 导入配置
from config_example import get_config

# 获取标准配置
config = get_config('standard')

# 获取交叉验证配置
cv_config = get_config('cross_validation')

# 获取快速训练配置（用于测试）
quick_config = get_config('quick')

# 获取高性能训练配置
high_perf_config = get_config('high_performance')

# 自定义配置参数
custom_config = get_config('standard', num_epochs=150, batch_size=2)
```

### 可用预设配置

- **quick**: 快速训练配置，适合测试代码
- **standard**: 标准训练配置，适合一般使用
- **cross_validation**: 交叉验证配置，适合小数据集
- **high_performance**: 高性能配置，适合正式训练
- **small_dataset**: 小数据集专用配置

### 基础配置参数

```python
config = {
    'data_dir': 'data',                    # 数据目录
    'num_epochs': 100,                     # 训练轮数
    'batch_size': 1,                       # 批次大小
    'learning_rate': 1e-4,                 # 学习率
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 训练设备
    'data_type': 'BPH',                    # 数据类型 ('BPH' 或 'PCA')
    'save_dir': 'checkpoints',             # 模型保存目录
    'handle_missing_modalities': 'zero_fill',  # 缺失模态处理策略
    'validation': True,                    # 是否使用验证集
}
```

### 交叉验证配置

```python
config = {
    # ... 基础配置 ...
    'n_splits': 5,                         # 交叉验证折数
    'handle_missing_modalities': 'zero_fill',  # 缺失模态处理
}
```

## 缺失模态处理策略

- **zero_fill**：使用零填充缺失的模态（默认）
- **skip**：跳过不完整的病例
- **duplicate**：使用参考模态填充

## 模型架构

使用3D U-Net架构，包含：
- 编码器路径：4个下采样层，逐步提取特征
- 解码器路径：4个上采样层，逐步恢复分辨率
- 跳跃连接：融合不同层级的特征
- 多模态输入：支持5种MRI模态同时输入

## 性能优化

- **GPU加速**：自动检测并使用CUDA
- **批归一化**：加速训练收敛
- **学习率调度**：ReduceLROnPlateau调度器
- **早停机制**：防止过拟合
- **权重初始化**：Kaiming初始化

## 注意事项

1. 对于小数据集（如240例），建议使用交叉验证版本以获得更稳定的性能评估
2. 确保所有模态数据的空间尺寸一致
3. 建议设置合适的批次大小以充分利用GPU内存
4. 训练过程中会定期保存检查点，可以从检查点恢复训练

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

MIT License