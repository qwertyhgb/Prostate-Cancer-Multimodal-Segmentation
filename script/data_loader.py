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
                 fold_indices=None, handle_missing_modalities='zero_fill'):
        """
        初始化前列腺数据集
        
        参数:
            root_dir (str): 数据集根目录
            mode (str): 数据模式，'train' 或 'test'
            transform: 数据增强转换函数
            data_type (str): 数据类型，'BPH' 或 'PCA'，默认为'BPH'
            fold_indices (list): 当前折的索引列表，用于交叉验证
            handle_missing_modalities (str): 处理缺失模态的方法
                - 'zero_fill': 使用零填充缺失模态
                - 'skip': 跳过不完整的病例
                - 'duplicate': 使用其他模态复制填充
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.data_type = data_type
        self.handle_missing_modalities = handle_missing_modalities
        
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
            
        # 根据处理策略过滤病例
        self.cases = self._filter_cases(self.cases)
        
        # 记录参考尺寸（用于统一图像尺寸）
        self.reference_shape = None
        
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
        
    def _filter_cases(self, cases):
        """根据处理策略过滤病例
        
        参数:
            cases (list): 原始病例列表
            
        返回:
            list: 过滤后的病例列表
        """
        if self.handle_missing_modalities == 'skip':
            # 跳过不完整的病例（包括模态和标签）
            filtered_cases = []
            for case_id in cases:
                # 检查所有模态是否存在
                modality_complete = True
                for modality in self.modalities:
                    filepath = os.path.join(self.root_dir, 'BPH-PCA', self.data_type, 
                                          modality, f"{case_id}.nii")
                    if not os.path.exists(filepath):
                        modality_complete = False
                        break
                
                # 检查标签是否存在
                label_path = os.path.join(self.root_dir, 'BPH-PCA', f'ROI（{self.data_type}+PCA）', 
                                         self.data_type, f"{case_id}.nii")
                label_exists = os.path.exists(label_path)
                
                if modality_complete and label_exists:
                    filtered_cases.append(case_id)
                    
            print(f"完整病例数: {len(filtered_cases)} / {len(cases)}")
            return filtered_cases
        else:
            # 其他策略过滤掉没有标签的病例
            filtered_cases = []
            for case_id in cases:
                label_path = os.path.join(self.root_dir, 'BPH-PCA', f'ROI（{self.data_type}+PCA）', 
                                         self.data_type, f"{case_id}.nii")
                if os.path.exists(label_path):
                    filtered_cases.append(case_id)
                    
            print(f"有标签的病例数: {len(filtered_cases)} / {len(cases)}")
            return filtered_cases
    
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
        
        try:
            # 使用SimpleITK读取图像
            # sitk.ReadImage可以读取多种医学图像格式，包括NIFTI
            # 使用兼容的编码方式处理中文路径
            sitk_image = sitk.ReadImage(filepath)
            
            # 将SimpleITK图像转换为numpy数组
            # SimpleITK图像的维度顺序通常是(Z, Y, X)，对应我们的(D, H, W)
            image_array = sitk.GetArrayFromImage(sitk_image)
            
            return image_array
        except Exception as e:
            print(f"加载文件时出错 {filepath}: {e}")
            # 如果文件存在但无法读取，创建一个默认的零数组
            return np.zeros((64, 64, 64), dtype=np.float32)
    
    def _resize_image(self, image, target_shape):
        """调整图像尺寸以匹配目标形状
        
        参数:
            image (numpy.ndarray): 输入图像
            target_shape (tuple): 目标形状 (D, H, W)
            
        返回:
            numpy.ndarray: 调整尺寸后的图像
        """
        # 如果形状已经匹配，直接返回
        if image.shape == target_shape:
            return image
            
        # 使用简单的插值方法调整尺寸
        from scipy.ndimage import zoom
        
        # 计算缩放因子
        zoom_factors = [t / s for t, s in zip(target_shape, image.shape)]
        
        # 调整图像尺寸
        resized_image = zoom(image, zoom_factors, order=1)  # 线性插值
        
        return resized_image
    
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
        reference_image = None  # 用于缺失模态的参考图像
        
        for i, modality in enumerate(self.modalities):
            # 构建图像文件路径
            filepath = os.path.join(self.root_dir, 'BPH-PCA', self.data_type, 
                                  modality, f"{case_id}.nii")
            
            # 检查文件是否存在
            if os.path.exists(filepath):
                # 使用SimpleITK加载图像
                try:
                    img = self._load_nifti_with_sitk(filepath)
                    # 保存第一个存在的模态作为参考图像
                    if reference_image is None:
                        reference_image = img
                except Exception as e:
                    print(f"加载模态 {modality} 时出错: {e}")
                    # 如果加载失败，使用零填充
                    if reference_image is not None:
                        img = np.zeros_like(reference_image, dtype=np.float32)
                    else:
                        img = np.zeros((64, 64, 64), dtype=np.float32)
            else:
                # 处理缺失的模态
                if self.handle_missing_modalities == 'zero_fill':
                    # 使用零填充
                    if reference_image is not None:
                        img = np.zeros_like(reference_image, dtype=np.float32)
                    else:
                        # 如果还没有参考图像，则创建一个默认大小的零图像
                        img = np.zeros((64, 64, 64), dtype=np.float32)
                elif self.handle_missing_modalities == 'duplicate':
                    # 使用参考图像填充（如果有的话）
                    if reference_image is not None:
                        img = reference_image.copy()
                    else:
                        # 如果还没有参考图像，则创建一个默认大小的零图像
                        img = np.zeros((64, 64, 64), dtype=np.float32)
                else:
                    # 其他情况抛出异常
                    raise FileNotFoundError(f"缺失模态文件且未指定处理策略: {filepath}")
            
            # 对图像进行预处理
            img = self._preprocess_image(img)
            # 添加到模态图像列表
            modality_images.append(img)
        
        # 设置参考形状（用于统一所有病例的图像尺寸）
        if self.reference_shape is None:
            self.reference_shape = reference_image.shape if reference_image is not None else (64, 64, 64)
        
        # 调整所有模态图像到统一尺寸
        resized_images = []
        for img in modality_images:
            if img.shape != self.reference_shape:
                img = self._resize_image(img, self.reference_shape)
            resized_images.append(img)
        
        # 将多个模态的图像堆叠在一起
        # axis=0表示在通道维度上堆叠，形成(5, D, H, W)的数组
        image = np.stack(resized_images, axis=0)
        
        # 加载分割标签
        # 标签文件存储在特定目录中
        label_path = os.path.join(self.root_dir, 'BPH-PCA', f'ROI（{self.data_type}+PCA）', 
                                 self.data_type, f"{case_id}.nii")
        
        try:
            # 加载标签图像
            label = self._load_nifti_with_sitk(label_path)
            # 调整标签图像到统一尺寸
            if label.shape != self.reference_shape:
                label = self._resize_image(label, self.reference_shape)
            # 转换为二值标签（前景为1，背景为0）
            label = (label > 0).astype(np.float32)
        except Exception as e:
            print(f"加载标签时出错 {label_path}: {e}")
            # 如果标签加载失败，创建一个全零标签
            label = np.zeros(self.reference_shape, dtype=np.float32)
        
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

def get_dataloader(root_dir, batch_size=1, num_workers=0, mode='train', data_type='BPH',
                   fold_indices=None, handle_missing_modalities='zero_fill'):
    """创建数据加载器
    
    参数:
        root_dir (str): 数据集根目录
        batch_size (int): 批次大小，默认为1
        num_workers (int): 数据加载的线程数，默认为0（避免Windows上的问题）
        mode (str): 数据模式，'train' 或 'test'
        data_type (str): 数据类型，'BPH' 或 'PCA'
        fold_indices (list): 当前折的索引列表，用于交叉验证
        handle_missing_modalities (str): 处理缺失模态的方法
            - 'zero_fill': 使用零填充缺失模态
            - 'skip': 跳过不完整的病例
            - 'duplicate': 使用其他模态复制填充
        
    返回:
        DataLoader: PyTorch数据加载器对象
    """
    # 创建前列腺数据集实例
    dataset = ProstateDataset(root_dir=root_dir, mode=mode, data_type=data_type,
                              fold_indices=fold_indices,
                              handle_missing_modalities=handle_missing_modalities)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        # 训练模式下打乱数据顺序，测试模式下保持顺序
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        # 将数据加载到GPU内存中以加速训练
        pin_memory=False  # 关闭pin_memory避免Windows上的问题
    )
    
    return dataloader

def get_kfold_splits(root_dir, data_type='BPH', n_splits=5, shuffle=True, random_state=42,
                     handle_missing_modalities='zero_fill'):
    """获取K折交叉验证的分割索引
    
    参数:
        root_dir (str): 数据集根目录
        data_type (str): 数据类型，'BPH' 或 'PCA'
        n_splits (int): 折数，默认为5
        shuffle (bool): 是否打乱数据，默认为True
        random_state (int): 随机种子，确保结果可重现，默认为42
        handle_missing_modalities (str): 处理缺失模态的方法
        
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
    
    # 如果需要跳过不完整的病例，先过滤
    if handle_missing_modalities == 'skip':
        filtered_cases = []
        modalities = ['ADC', 'DWI', 'gaoqing-T2', 'T2 fs', 'T2 not fs']
        for case_id in all_cases:
            complete = True
            for modality in modalities:
                filepath = os.path.join(root_dir, 'BPH-PCA', data_type, 
                                      modality, f"{case_id}.nii")
                if not os.path.exists(filepath):
                    complete = False
                    break
            if complete:
                filtered_cases.append(case_id)
        all_cases = filtered_cases
        n_samples = len(all_cases)
        print(f"交叉验证将使用 {n_samples} 个完整病例")
    
    # 创建K折交叉验证分割器
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    # 生成分割索引
    splits = []
    for train_indices, val_indices in kf.split(range(n_samples)):
        # 将numpy数组转换为列表
        splits.append((train_indices.tolist(), val_indices.tolist()))
    
    return splits