# 前列腺多模态MRI图像分割项目

本项目使用3D U-Net模型对前列腺多模态MRI图像进行分割，支持BPH（良性前列腺增生）和PCA（前列腺癌）两种类型的数据。

## 项目结构

```
U-Net_3D/
├── models/                  # 模型定义
│   └── unet3d.py           # 3D U-Net模型实现
├── script/                  # 数据处理和辅助脚本
│   ├── data_loader.py      # 数据加载器
│   ├── validate_model.py   # 模型验证脚本
│   └── predict.py          # 模型预测脚本
├── train.py                # 基础训练脚本
├── train_bph.py            # BPH数据专用训练脚本
├── train_bph_cv.py         # BPH数据交叉验证训练脚本
├── data/                   # 数据目录
│   └── BPH-PCA/           # BPH和PCA数据
│       ├── BPH/           # BPH数据
│       │   ├── ADC/       # ADC模态图像
│       │   ├── DWI/       # DWI模态图像
│       │   ├── T2 fs/     # T2脂肪抑制模态图像
│       │   ├── T2 not fs/ # T2非脂肪抑制模态图像
│       │   └── gaoqing-T2/ # 高清T2模态图像
│       ├── PCA/           # PCA数据
│       └── ROI（BPH+PCA）/ # 分割标签
└── checkpoints/            # 模型检查点
```

## 功能介绍

### 1. 模型训练

#### 基础训练
```bash
python train_bph.py
```

#### 交叉验证训练（推荐用于小数据集）
```bash
python train_bph_cv.py
```

### 2. 模型验证
```bash
python script/validate_model.py
```

### 3. 模型预测
```bash
python script/predict.py
```

## 数据格式

数据应按照以下格式组织：

```
BPH-PCA/
├── BPH/
│   ├── ADC/
│   │   ├── case_001.nii
│   │   ├── case_002.nii
│   │   └── ...
│   ├── DWI/
│   ├── T2 fs/
│   ├── T2 not fs/
│   └── gaoqing-T2/
└── ROI（BPH+PCA）/
    └── BPH/
        ├── case_001.nii
        ├── case_002.nii
        └── ...
```

## 使用说明

### 训练模型

1. 修改训练脚本中的配置参数
2. 运行训练脚本

### 验证模型

1. 修改`validate_model.py`中的配置参数，指定模型路径
2. 运行验证脚本

### 预测新数据

1. 准备新数据，确保目录结构正确
2. 修改`predict.py`中的配置参数
3. 运行预测脚本

## 依赖项

- PyTorch
- SimpleITK
- tqdm
- scikit-learn
- numpy

## 特点

1. **多模态支持**：支持ADC、DWI、T2 fs、T2 not fs、gaoqing-T2五种模态
2. **3D U-Net架构**：专为3D医学图像分割设计
3. **交叉验证**：针对小数据集（如240例）优化的5折交叉验证
4. **完整的评估流程**：包含Dice系数和IoU等多种评估指标
5. **详细的中文注释**：便于理解和维护