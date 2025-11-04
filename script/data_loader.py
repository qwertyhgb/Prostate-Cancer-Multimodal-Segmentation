import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from sklearn.model_selection import KFold

class ProstateDataset(Dataset):
    """前列腺多模态MRI数据集类
    
    专门用于加载和处理前列腺多模态MRI数据
    支持5种模态：ADC、DWI、gaoqing-T2、T2 fs、T2 not fs
    针对BPH和PCA数据进行了优化
    
    主要功能：
    - 自动扫描和加载多模态数据
    - 处理缺失模态情况
    - 图像预处理和标准化
    - 标签加载和二值化
    - 支持交叉验证数据划分
    
    数据组织：
    - 每个病例对应一个文件夹
    - 每种模态对应一个NIFTI文件
    - 标签文件为segmentation.nii.gz
    """
    
    def __init__(self, data_dir, modalities=None, missing_strategy='zero_fill', 
                 target_size=(128, 128, 128), is_training=True, data_type='BPH'):
        """
        初始化数据集
        
        参数:
            data_dir (str): 数据根目录路径
            modalities (list): 模态列表，默认包含5种标准模态
            missing_strategy (str): 缺失模态处理策略
                - 'zero_fill': 用零填充缺失模态
                - 'skip': 跳过缺失模态的病例
                - 'duplicate': 用其他模态复制填充
            target_size (tuple): 目标图像尺寸 (D, H, W)
            is_training (bool): 是否为训练模式
            data_type (str): 数据类型 ('BPH' 或 'PCA')
        """
        self.data_dir = data_dir
        # 设置默认模态列表：ADC、DWI、高分辨率T2、T2脂肪抑制、T2非脂肪抑制
        self.modalities = modalities or ['ADC', 'DWI', 'gaoqing-T2', 'T2 fs', 'T2 not fs']
        self.missing_strategy = missing_strategy
        self.target_size = target_size
        self.is_training = is_training
        self.data_type = data_type
        
        # 获取病例列表并过滤有效病例
        self.case_list = self._get_case_list()
        self.case_list = self._filter_cases()
        
        print(f"数据集初始化完成，共找到 {len(self.case_list)} 个有效病例")
        print(f"数据类型: {data_type}")
        print(f"模态列表: {self.modalities}")
        print(f"缺失模态处理策略: {missing_strategy}")

    def _get_case_list(self):
        """
        扫描数据目录，获取所有病例文件夹列表
        
        返回:
            list: 病例文件夹名称列表
            
        扫描规则：
        - 只扫描直接子目录
        - 排除隐藏文件和系统文件
        - 每个病例对应一个独立的文件夹
        - 根据data_type参数过滤病例
        """
        # 获取数据目录下的所有子目录
        all_items = os.listdir(self.data_dir)
        # 过滤出目录项，排除文件
        case_dirs = [item for item in all_items 
                    if os.path.isdir(os.path.join(self.data_dir, item))]
        # 排除隐藏文件和系统文件
        case_dirs = [case for case in case_dirs if not case.startswith('.')]
        
        # 根据data_type参数过滤病例
        if self.data_type == 'BPH':
            # 对于BPH数据，我们假设病例文件夹名包含'BPH'或不包含'PCA'
            filtered_cases = [case for case in case_dirs if 'BPH' in case.upper() or 'PCA' not in case.upper()]
        elif self.data_type == 'PCA':
            # 对于PCA数据，我们假设病例文件夹名包含'PCA'
            filtered_cases = [case for case in case_dirs if 'PCA' in case.upper()]
        else:
            # 默认情况下，返回所有病例
            filtered_cases = case_dirs
        
        print(f"扫描到 {len(case_dirs)} 个病例文件夹，根据数据类型 '{self.data_type}' 过滤后剩余 {len(filtered_cases)} 个")
        return filtered_cases

    def _filter_cases(self):
        """
        过滤病例，只保留包含必要模态数据的病例
        
        返回:
            list: 过滤后的有效病例列表
            
        过滤标准：
        - 必须包含标签文件 (segmentation.nii.gz)
        - 根据缺失模态策略决定是否保留
        - 检查模态文件的完整性和可读性
        """
        valid_cases = []
        
        for case in self.case_list:
            case_path = os.path.join(self.data_dir, case)
            
            # 检查标签文件是否存在
            label_path = os.path.join(case_path, 'segmentation.nii.gz')
            if not os.path.exists(label_path):
                print(f"警告: 病例 {case} 缺少标签文件，已跳过")
                continue
            
            # 检查模态文件
            modality_files = {}
            missing_modalities = []
            
            for modality in self.modalities:
                # 构建模态文件路径模式
                pattern = os.path.join(case_path, f"*{modality}*.nii*")
                files = glob.glob(pattern)
                
                if files:
                    # 取第一个匹配的文件
                    modality_files[modality] = files[0]
                else:
                    missing_modalities.append(modality)
            
            # 根据缺失模态策略处理
            if missing_modalities:
                if self.missing_strategy == 'skip':
                    print(f"警告: 病例 {case} 缺失模态 {missing_modalities}，已跳过")
                    continue
                elif self.missing_strategy == 'duplicate':
                    # 用第一个可用模态复制填充缺失模态
                    available_modalities = [m for m in self.modalities if m not in missing_modalities]
                    if available_modalities:
                        duplicate_modality = available_modalities[0]
                        for missing_mod in missing_modalities:
                            modality_files[missing_mod] = modality_files[duplicate_modality]
                        print(f"信息: 病例 {case} 缺失模态 {missing_modalities}，已用 {duplicate_modality} 复制填充")
                    else:
                        print(f"警告: 病例 {case} 所有模态都缺失，已跳过")
                        continue
                # 'zero_fill' 策略在加载时处理
            
            # 检查模态文件是否可读
            try:
                for modality, file_path in modality_files.items():
                    # 尝试读取文件头信息验证文件完整性
                    reader = sitk.ImageFileReader()
                    reader.SetFileName(file_path)
                    reader.ReadImageInformation()
            except Exception as e:
                print(f"警告: 病例 {case} 的模态文件读取失败: {e}，已跳过")
                continue
            
            # 记录有效病例及其模态文件信息
            valid_cases.append({
                'case_id': case,
                'case_path': case_path,
                'modality_files': modality_files,
                'label_path': label_path,
                'missing_modalities': missing_modalities
            })
        
        return valid_cases

    def _load_nifti_with_sitk(self, file_path):
        """
        使用SimpleITK加载NIFTI格式图像
        
        参数:
            file_path (str): NIFTI文件路径
            
        返回:
            numpy.ndarray: 图像数据数组
            
        处理步骤：
        - 读取NIFTI文件
        - 转换为numpy数组
        - 处理可能的维度问题
        - 确保数据类型一致性
        """
        try:
            # 使用SimpleITK读取图像
            image = sitk.ReadImage(file_path)
            # 转换为numpy数组
            array = sitk.GetArrayFromImage(image)
            
            # 处理单通道图像，确保维度一致性
            if len(array.shape) == 3:
                # 3D图像，添加通道维度
                array = array[np.newaxis, ...]  # 形状: (1, D, H, W)
            elif len(array.shape) == 4:
                # 4D图像，可能是多通道或多时间点
                # 取第一个通道/时间点
                array = array[0:1, ...]
            else:
                raise ValueError(f"不支持的图像维度: {array.shape}")
            
            return array
            
        except Exception as e:
            print(f"错误: 加载文件 {file_path} 失败: {e}")
            # 根据缺失模态策略返回相应数据
            if self.missing_strategy == 'zero_fill':
                # 返回零填充的图像
                return np.zeros((1,) + self.target_size, dtype=np.float32)
            else:
                raise

    def _preprocess_image(self, image_array):
        """
        图像预处理
        
        参数:
            image_array (numpy.ndarray): 原始图像数组
            
        返回:
            numpy.ndarray: 预处理后的图像数组
            
        预处理步骤：
        - 尺寸调整到目标大小
        - 强度归一化到[0,1]范围
        - 数据类型转换
        - 异常值处理
        """
        # 确保数据类型为float32
        image_array = image_array.astype(np.float32)
        
        # 尺寸调整
        if image_array.shape[1:] != self.target_size:
            # 使用SimpleITK进行重采样
            original_image = sitk.GetImageFromArray(image_array[0])
            
            # 计算重采样参数
            original_size = original_image.GetSize()
            target_size = self.target_size
            
            # 创建重采样器
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(target_size)
            resampler.SetOutputSpacing([
                original_size[0] * original_image.GetSpacing()[0] / target_size[0],
                original_size[1] * original_image.GetSpacing()[1] / target_size[1],
                original_size[2] * original_image.GetSpacing()[2] / target_size[2]
            ])
            resampler.SetOutputOrigin(original_image.GetOrigin())
            resampler.SetOutputDirection(original_image.GetDirection())
            resampler.SetInterpolator(sitk.sitkLinear)
            
            # 执行重采样
            resampled_image = resampler.Execute(original_image)
            image_array = sitk.GetArrayFromImage(resampled_image)
            image_array = image_array[np.newaxis, ...]  # 重新添加通道维度
        
        # 强度归一化
        # 排除零值区域进行归一化
        nonzero_mask = image_array != 0
        if np.any(nonzero_mask):
            nonzero_values = image_array[nonzero_mask]
            # 使用非零值的百分位数进行归一化，避免异常值影响
            p1, p99 = np.percentile(nonzero_values, [1, 99])
            image_array = np.clip(image_array, p1, p99)
            image_array = (image_array - p1) / (p99 - p1 + 1e-8)
        
        return image_array

    def __len__(self):
        """
        返回数据集中的病例数量
        
        返回:
            int: 病例数量
        """
        return len(self.case_list)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        
        参数:
            idx (int): 样本索引
            
        返回:
            dict: 包含图像数据和标签的字典
                - 'image': 多模态图像张量，形状为(n_modalities, D, H, W)
                - 'label': 分割标签张量，形状为(1, D, H, W)
                - 'case_id': 病例标识符
        """
        case_info = self.case_list[idx]
        case_id = case_info['case_id']
        
        # 加载多模态图像数据
        modality_images = []
        
        for modality in self.modalities:
            if modality in case_info['modality_files']:
                # 加载存在的模态
                file_path = case_info['modality_files'][modality]
                image_array = self._load_nifti_with_sitk(file_path)
            else:
                # 处理缺失模态
                if self.missing_strategy == 'zero_fill':
                    # 零填充
                    image_array = np.zeros((1,) + self.target_size, dtype=np.float32)
                elif self.missing_strategy == 'duplicate':
                    # 复制其他模态（在过滤阶段已处理）
                    available_modalities = [m for m in self.modalities 
                                           if m in case_info['modality_files']]
                    if available_modalities:
                        duplicate_file = case_info['modality_files'][available_modalities[0]]
                        image_array = self._load_nifti_with_sitk(duplicate_file)
                    else:
                        image_array = np.zeros((1,) + self.target_size, dtype=np.float32)
                else:
                    raise ValueError(f"不支持的缺失模态策略: {self.missing_strategy}")
            
            # 预处理图像
            image_array = self._preprocess_image(image_array)
            modality_images.append(image_array)
        
        # 合并多模态数据
        # 形状: (n_modalities, D, H, W)
        multimodal_image = np.concatenate(modality_images, axis=0)
        
        # 加载标签数据
        label_array = self._load_nifti_with_sitk(case_info['label_path'])
        # 标签预处理：二值化
        label_array = (label_array > 0).astype(np.float32)
        
        # 转换为PyTorch张量
        image_tensor = torch.from_numpy(multimodal_image).float()
        label_tensor = torch.from_numpy(label_array).float()
        
        return {
            'image': image_tensor,
            'label': label_tensor,
            'case_id': case_id
        }

def get_dataloader(data_dir, batch_size=2, shuffle=True, modalities=None, 
                   missing_strategy='zero_fill', target_size=(128, 128, 128), 
                   num_workers=4, is_training=True, data_type='BPH', indices=None):
    """
    创建数据加载器
    
    参数:
        data_dir (str): 数据目录路径
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据
        modalities (list): 模态列表
        missing_strategy (str): 缺失模态处理策略
        target_size (tuple): 目标图像尺寸
        num_workers (int): 数据加载工作进程数
        is_training (bool): 是否为训练模式
        data_type (str): 数据类型 ('BPH' 或 'PCA')
        indices (list): 指定使用的数据索引列表（用于交叉验证）
        
    返回:
        DataLoader: PyTorch数据加载器
    """
    # 创建数据集实例
    dataset = ProstateDataset(
        data_dir=data_dir,
        modalities=modalities,
        missing_strategy=missing_strategy,
        target_size=target_size,
        is_training=is_training,
        data_type=data_type
    )
    
    # 如果指定了索引，则使用Subset创建子数据集
    if indices is not None:
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # 加速GPU数据传输
    )
    
    return dataloader

def get_kfold_splits(data_dir, n_splits=5, modalities=None, 
                    missing_strategy='zero_fill', target_size=(128, 128, 128), data_type='BPH'):
    """
    生成K折交叉验证的数据划分
    
    参数:
        data_dir (str): 数据目录路径
        n_splits (int): 交叉验证折数
        modalities (list): 模态列表
        missing_strategy (str): 缺失模态处理策略
        target_size (tuple): 目标图像尺寸
        data_type (str): 数据类型 ('BPH' 或 'PCA')
        
    返回:
        list: 包含K个(train_indices, val_indices)元组的列表
    """
    # 创建数据集实例以获取病例列表
    dataset = ProstateDataset(
        data_dir=data_dir,
        modalities=modalities,
        missing_strategy=missing_strategy,
        target_size=target_size,
        data_type=data_type
    )
    
    # 获取病例数量
    n_cases = len(dataset)
    
    # 使用KFold进行数据划分
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    splits = []
    for train_indices, val_indices in kfold.split(range(n_cases)):
        splits.append((train_indices, val_indices))
    
    print(f"生成 {n_splits} 折交叉验证划分，共 {n_cases} 个病例")
    return splits