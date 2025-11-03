import os
import torch
import numpy as np
from models.unet3d import UNet3D
import SimpleITK as sitk
from datetime import datetime

def load_multimodal_images(case_dir):
    """加载多模态MRI图像
    
    从指定目录加载五种模态的MRI图像数据
    
    参数:
        case_dir (str): 病例数据目录，应包含ADC, DWI, T2 fs, T2 not fs, gaoqing-T2等子目录
        
    返回:
        tuple: (images, modality_names) 其中images是numpy数组，modality_names是模态名称列表
        
    异常:
        FileNotFoundError: 当目录或文件不存在时抛出
    """
    # 定义MRI模态列表
    modalities = ['ADC', 'DWI', 'gaoqing-T2', 'T2 fs', 'T2 not fs']
    modality_images = []
    
    # 遍历每种模态
    for modality in modalities:
        # 构建模态目录路径
        modality_dir = os.path.join(case_dir, modality)
        if not os.path.exists(modality_dir):
            raise FileNotFoundError(f"模态目录不存在: {modality_dir}")
            
        # 获取该模态下的.nii文件
        nii_files = [f for f in os.listdir(modality_dir) if f.endswith('.nii')]
        if not nii_files:
            raise FileNotFoundError(f"在 {modality_dir} 中未找到.nii文件")
            
        # 如果找到多个.nii文件，给出警告并使用第一个文件
        if len(nii_files) > 1:
            print(f"警告: 在 {modality_dir} 中找到多个.nii文件，将使用第一个文件")
            
        # 加载图像
        filepath = os.path.join(modality_dir, nii_files[0])
        sitk_image = sitk.ReadImage(filepath)
        image_array = sitk.GetArrayFromImage(sitk_image)
        
        # 预处理图像数据
        image_array = image_array.astype(np.float32)
        # 归一化到[0,1]范围
        if image_array.max() - image_array.min() != 0:
            image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        else:
            image_array = np.zeros_like(image_array, dtype=np.float32)
            
        # 添加到模态图像列表
        modality_images.append(image_array)
    
    # 将多个模态的图像堆叠在一起，形成(5, D, H, W)的数组
    image = np.stack(modality_images, axis=0)
    return image, modalities

def preprocess_image(image):
    """预处理图像数据
    
    将numpy数组转换为PyTorch张量并添加批次维度
    
    参数:
        image (numpy.ndarray): 原始图像数据，形状为(5, D, H, W)
        
    返回:
        torch.Tensor: 预处理后的图像张量，形状为(1, 5, D, H, W)
    """
    # 转换为PyTorch张量
    image_tensor = torch.from_numpy(image).float()
    
    # 添加批次维度，形成(1, 5, D, H, W)的张量
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

class ModelPredictor:
    """模型预测器
    
    用于对新的MRI数据进行前列腺分割预测
    """
    
    def __init__(self, model_path, device='cpu'):
        """
        初始化模型预测器
        
        参数:
            model_path (str): 模型文件路径
            device (str): 设备 ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        
        # 加载模型
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """加载训练好的模型
        
        参数:
            model_path (str): 模型文件路径
            
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
        checkpoint = torch.load(model_path, map_location=self.device)
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
        
    def predict(self, image_tensor):
        """进行预测
        
        参数:
            image_tensor (torch.Tensor): 输入图像张量，形状为(1, 5, D, H, W)
            
        返回:
            numpy.ndarray: 预测结果，形状为(D, H, W)
        """
        print("正在进行预测...")
        # 关闭梯度计算
        with torch.no_grad():
            # 将图像张量移动到指定设备
            image_tensor = image_tensor.to(self.device)
            # 模型预测
            output = self.model.predict(image_tensor)
            # 去除批次和类别维度，得到(D, H, W)的预测结果
            prediction = output.squeeze(0).squeeze(0).cpu().numpy()
            
        print("预测完成!")
        return prediction
        
    def save_prediction(self, prediction, output_path, reference_image_path=None):
        """保存预测结果
        
        参数:
            prediction (numpy.ndarray): 预测结果，形状为(D, H, W)
            output_path (str): 输出文件路径
            reference_image_path (str): 参考图像路径，用于保持相同的元数据
        """
        print(f"正在保存预测结果至: {output_path}")
        
        # 转换为二值分割结果（阈值为0.5）
        binary_prediction = (prediction > 0.5).astype(np.uint8)
        
        # 创建SimpleITK图像
        sitk_image = sitk.GetImageFromArray(binary_prediction)
        
        # 如果提供了参考图像，则复制其元数据（空间信息等）
        if reference_image_path and os.path.exists(reference_image_path):
            reference_sitk = sitk.ReadImage(reference_image_path)
            sitk_image.CopyInformation(reference_sitk)
            
        # 保存图像为NIFTI格式
        sitk.WriteImage(sitk_image, output_path)
        print("预测结果保存完成!")

def main():
    """主函数"""
    # 预测配置参数
    config = {
        'model_path': '',  # 需要指定模型文件路径
        'case_dir': '',    # 需要指定病例数据目录
        'output_dir': os.path.join('results', f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 检查必要参数
    if not config['model_path'] or not os.path.exists(config['model_path']):
        print("请在config中指定有效的模型文件路径!")
        print("例如: config['model_path'] = 'checkpoints/BPH_20240101_120000/best_model_epoch_50.pth'")
        return
        
    if not config['case_dir'] or not os.path.exists(config['case_dir']):
        print("请在config中指定有效的病例数据目录!")
        print("例如: config['case_dir'] = 'data/new_cases/case_001'")
        return
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 创建预测器
    predictor = ModelPredictor(config['model_path'], config['device'])
    
    # 加载多模态图像
    print("正在加载多模态MRI图像...")
    image, modalities = load_multimodal_images(config['case_dir'])
    print(f"已加载 {len(modalities)} 个模态: {', '.join(modalities)}")
    print(f"图像形状: {image.shape}")
    
    # 预处理图像
    image_tensor = preprocess_image(image)
    
    # 进行预测
    prediction = predictor.predict(image_tensor)
    print(f"预测结果形状: {prediction.shape}")
    
    # 保存预测结果
    output_path = os.path.join(config['output_dir'], 'prediction.nii')
    
    # 使用第一个模态的图像作为参考来保持元数据
    reference_modality = modalities[0]
    reference_image_path = os.path.join(config['case_dir'], reference_modality, 
                                       os.listdir(os.path.join(config['case_dir'], reference_modality))[0])
    
    predictor.save_prediction(prediction, output_path, reference_image_path)
    print(f"所有结果已保存至: {config['output_dir']}")

if __name__ == '__main__':
    main()