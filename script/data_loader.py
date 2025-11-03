import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from sklearn.model_selection import KFold

class ProstateDataset(Dataset):
    """前列腺多模态MRI数据集
    
    用于加载和预处理前列腺多模态MRI图像和对应的分割标签
    专门针对BPH数据进行优化
    """
    def __init__(self, root_dir, mode='train', transform=None, data_type='BPH', 
                 fold_indices=None):
        """
        初始化前列腺数据集
        
        参数:
            root_dir (str): 数据集根目录
            mode (str): 数据模式，'train' 或 'test'
            transform: 数据增强转换函数
            data_type (str): 数据类型，'BPH' 或 'PCA'，默认为'BPH'
            fold_indices (list): 当前折的索引列表，用于交叉验证
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.data_type = data_type  # 指定数据类型(BPH或PCA)
        
        # 定义MRI模态列表
        # 包括ADC、DWI、高清T2、T2脂肪抑制和T2非脂肪抑制五种模态
        self.modalities = ['ADC', 'DWI', 'gaoqing-T2', 'T2 fs', 'T2 not fs']
        
        # 获取所有病例ID列表
        all_cases = self._get_case_list()
        
        # 如果提供了折索引，则使用这些索引选择病例（用于交叉验证）
        if fold_indices is not None:
            self.cases = [all_cases[i] for i in fold_indices]
        else:
            self.cases = all_cases
        
    def _get_case_list(self):
        """获取所有可用的病例ID
        
        通过扫描ADC模态目录来获取所有病例ID列表
        选择ADC模态是因为所有病例都应包含此模态
        
        返回:
            list: 排序后的病例ID列表
        """
        # 构建ADC模态数据目录路径
        adc_dir = os.path.join(self.root_dir, 'BPH-PCA', self.data_type, 'ADC')
        # 检查目录是否存在
        if not os.path.exists(adc_dir):
            raise FileNotFoundError(f"数据目录不存在: {adc_dir}")
        
        # 获取所有.nii文件的文件名（不含扩展名）
        # 这些文件名即为病例ID
        cases = [f.split('.')[0] for f in os.listdir(adc_dir) if f.endswith('.nii')]
        # 对病例ID进行排序以确保一致性
        return sorted(cases)
    
    def _load_nifti_with_sitk(self, filepath):
        """使用SimpleITK加载NIFTI格式的图像数据
        
        参数:
            filepath (str): NIFTI文件路径
            
        返回:
            numpy.ndarray: 图像数据数组，形状为(D, H, W)
            
        异常:
            FileNotFoundError: 当文件不存在时抛出
        """
        # 检查文件是否存在
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        # 使用SimpleITK读取图像
        # sitk.ReadImage可以读取多种医学图像格式，包括NIFTI
        sitk_image = sitk.ReadImage(filepath)
        
        # 将SimpleITK图像转换为numpy数组
        # SimpleITK图像的维度顺序通常是(Z, Y, X)，对应我们的(D, H, W)
        image_array = sitk.GetArrayFromImage(sitk_image)
        
        return image_array
    
    def _preprocess_image(self, image):
        """预处理图像数据
        
        对图像进行归一化处理，将像素值缩放到[0,1]范围
        
        参数:
            image (numpy.ndarray): 原始图像数据
            
        返回:
            numpy.ndarray: 预处理后的图像数据
        """
        # 转换为浮点型以支持小数运算
        image = image.astype(np.float32)
        
        # 标准化到[0,1]范围
        # 防止除零错误
        if image.max() - image.min() != 0:
            # 将像素值缩放到[0,1]范围
            image = (image - image.min()) / (image.max() - image.min())
        else:
            # 如果所有像素值相同，则设置为0
            image = np.zeros_like(image, dtype=np.float32)
        
        return image
    
    def __len__(self):
        """
        获取数据集大小
        
        返回:
            int: 数据集中样本的数量
        """
        return len(self.cases)
    
    def __getitem__(self, idx):
        """获取一个数据样本
        
        参数:
            idx (int): 样本索引
            
        返回:
            dict: 包含多模态图像、标签和病例ID的字典
                - 'image': 多模态图像张量，形状为(5, D, H, W)
                - 'label': 分割标签张量，形状为(D, H, W)
                - 'case_id': 病例ID
        """
        # 根据索引获取病例ID
        case_id = self.cases[idx]
        
        # 加载多模态图像
        modality_images = []
        for modality in self.modalities:
            # 构建图像文件路径
            filepath = os.path.join(self.root_dir, 'BPH-PCA', self.data_type, 
                                  modality, f"{case_id}.nii")
            # 使用SimpleITK加载图像
            img = self._load_nifti_with_sitk(filepath)
            # 对图像进行预处理
            img = self._preprocess_image(img)
            # 添加到模态图像列表
            modality_images.append(img)
        
        # 将多个模态的图像堆叠在一起
        # axis=0表示在通道维度上堆叠，形成(5, D, H, W)的数组
        image = np.stack(modality_images, axis=0)
        
        # 加载分割标签
        # 标签文件存储在特定目录中
        label_path = os.path.join(self.root_dir, 'BPH-PCA', f'ROI（{self.data_type}+PCA）', 
                                 self.data_type, f"{case_id}.nii")
        # 加载标签图像
        label = self._load_nifti_with_sitk(label_path)
        # 转换为二值标签（前景为1，背景为0）
        label = (label > 0).astype(np.float32)
        
        # 转换为PyTorch张量
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        
        # 应用数据增强（如果指定了转换函数）
        if self.transform:
            image, label = self.transform(image, label)
            
        # 返回包含图像、标签和病例ID的字典
        return {
            'image': image,
            'label': label,
            'case_id': case_id
        }

def get_dataloader(root_dir, batch_size=1, num_workers=4, mode='train', data_type='BPH',
                   fold_indices=None):
    """创建数据加载器
    
    参数:
        root_dir (str): 数据集根目录
        batch_size (int): 批次大小，默认为1
        num_workers (int): 数据加载的线程数，默认为4
        mode (str): 数据模式，'train' 或 'test'
        data_type (str): 数据类型，'BPH' 或 'PCA'
        fold_indices (list): 当前折的索引列表，用于交叉验证
        
    返回:
        DataLoader: PyTorch数据加载器对象
    """
    # 创建前列腺数据集实例
    dataset = ProstateDataset(root_dir=root_dir, mode=mode, data_type=data_type,
                              fold_indices=fold_indices)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        # 训练模式下打乱数据顺序，测试模式下保持顺序
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        # 将数据加载到GPU内存中以加速训练
        pin_memory=True
    )
    
    return dataloader

def get_kfold_splits(root_dir, data_type='BPH', n_splits=5, shuffle=True, random_state=42):
    """获取K折交叉验证的分割索引
    
    参数:
        root_dir (str): 数据集根目录
        data_type (str): 数据类型，'BPH' 或 'PCA'
        n_splits (int): 折数，默认为5
        shuffle (bool): 是否打乱数据，默认为True
        random_state (int): 随机种子，确保结果可重现，默认为42
        
    返回:
        list: 每个元素是一个元组(train_indices, val_indices)，包含训练和验证索引
        
    异常:
        FileNotFoundError: 当数据目录不存在时抛出
    """
    # 获取所有病例ID
    adc_dir = os.path.join(root_dir, 'BPH-PCA', data_type, 'ADC')
    if not os.path.exists(adc_dir):
        raise FileNotFoundError(f"数据目录不存在: {adc_dir}")
    
    # 获取所有.nii文件的文件名（不含扩展名）
    all_cases = [f.split('.')[0] for f in os.listdir(adc_dir) if f.endswith('.nii')]
    n_samples = len(all_cases)
    
    # 创建K折交叉验证分割器
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    # 生成分割索引
    splits = []
    for train_indices, val_indices in kf.split(range(n_samples)):
        # 将numpy数组转换为列表
        splits.append((train_indices.tolist(), val_indices.tolist()))
    
    return splits