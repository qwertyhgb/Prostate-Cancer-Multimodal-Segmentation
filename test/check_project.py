#!/usr/bin/env python3
"""
前列腺多模态MRI分割项目 - 项目状态检查脚本

这个脚本用于检查项目的完整性，包括文件结构、依赖安装、模型状态等。
"""

import os
import sys
import importlib
from datetime import datetime
import json

def check_file_structure():
    """检查项目文件结构"""
    print("=" * 60)
    print("检查项目文件结构...")
    print("=" * 60)
    
    # 必需的文件和目录
    required_files = {
        'models/unet3d.py': '3D U-Net模型实现',
        'script/data_loader.py': '数据加载器',
        'utils/losses.py': '损失函数模块',
        'utils/trainer.py': '训练器模块',
        'train_bph.py': 'BPH训练脚本',
        'train_bph_cv.py': '交叉验证训练脚本',
        'train_bph_optimized.py': '优化版训练脚本',
        'config_example.py': '配置文件示例',
        'run.py': '统一运行脚本',
        'requirements.txt': '项目依赖',
        'README_OPTIMIZED.md': '项目文档'
    }
    
    # 可选文件
    optional_files = {
        'script/validate_model.py': '模型验证脚本',
        'script/predict.py': '模型预测脚本',
        '.vscode/settings.json': 'VSCode配置',
        'data/': '数据目录',
        'checkpoints/': '模型检查点目录'
    }
    
    missing_required = []
    missing_optional = []
    
    print("必需文件检查:")
    for file_path, description in required_files.items():
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path) if os.path.isfile(full_path) else "目录"
            print(f"  ✓ {file_path} ({description}) - 大小: {size} bytes")
        else:
            print(f"  ✗ {file_path} ({description}) - 缺失")
            missing_required.append(file_path)
    
    print("\n可选文件检查:")
    for file_path, description in optional_files.items():
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            if os.path.isfile(full_path):
                size = os.path.getsize(full_path)
                print(f"  ✓ {file_path} ({description}) - 大小: {size} bytes")
            else:
                print(f"  ✓ {file_path} ({description}) - 目录存在")
        else:
            print(f"  - {file_path} ({description}) - 缺失 (可选)")
            missing_optional.append(file_path)
    
    # 总结
    print(f"\n文件结构检查结果:")
    print(f"  必需文件: {len(required_files) - len(missing_required)}/{len(required_files)} 个存在")
    print(f"  可选文件: {len(optional_files) - len(missing_optional)}/{len(optional_files)} 个存在")
    
    if missing_required:
        print(f"  缺失必需文件: {', '.join(missing_required)}")
        return False
    else:
        print("  所有必需文件都存在 ✓")
        return True

def check_dependencies():
    """检查项目依赖"""
    print("\n" + "=" * 60)
    print("检查项目依赖...")
    print("=" * 60)
    
    # 必需依赖
    required_deps = [
        'torch', 'torchvision', 'numpy', 'tqdm', 
        'SimpleITK', 'sklearn', 'matplotlib'
    ]
    
    # 可选依赖
    optional_deps = [
        'nibabel', 'seaborn', 'pandas', 'tensorboard',
        'pytest', 'black', 'flake8'
    ]
    
    missing_required = []
    missing_optional = []
    
    print("必需依赖检查:")
    for dep in required_deps:
        try:
            module = importlib.import_module(dep)
            if hasattr(module, '__version__'):
                version = module.__version__
                print(f"  ✓ {dep} (版本: {version})")
            else:
                print(f"  ✓ {dep} (版本: 未知)")
        except ImportError:
            print(f"  ✗ {dep} - 未安装")
            missing_required.append(dep)
    
    print("\n可选依赖检查:")
    for dep in optional_deps:
        try:
            module = importlib.import_module(dep)
            if hasattr(module, '__version__'):
                version = module.__version__
                print(f"  ✓ {dep} (版本: {version})")
            else:
                print(f"  ✓ {dep} (版本: 未知)")
        except ImportError:
            print(f"  - {dep} - 未安装 (可选)")
            missing_optional.append(dep)
    
    # 总结
    print(f"\n依赖检查结果:")
    print(f"  必需依赖: {len(required_deps) - len(missing_required)}/{len(required_deps)} 个已安装")
    print(f"  可选依赖: {len(optional_deps) - len(missing_optional)}/{len(optional_deps)} 个已安装")
    
    if missing_required:
        print(f"  缺失必需依赖: {', '.join(missing_required)}")
        print(f"  安装命令: pip install {' '.join(missing_required)}")
        return False
    else:
        print("  所有必需依赖都已安装 ✓")
        return True

def check_model_files():
    """检查模型文件"""
    print("\n" + "=" * 60)
    print("检查模型文件...")
    print("=" * 60)
    
    checkpoints_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    
    if not os.path.exists(checkpoints_dir):
        print("  检查点目录不存在")
        print("  如需训练模型，请先创建数据目录并准备数据")
        return True
    
    model_files = []
    for file in os.listdir(checkpoints_dir):
        if file.endswith('.pth'):
            file_path = os.path.join(checkpoints_dir, file)
            size = os.path.getsize(file_path)
            modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            model_files.append({
                'name': file,
                'size': size,
                'modified': modified_time
            })
    
    if model_files:
        print(f"  找到 {len(model_files)} 个模型文件:")
        for model in sorted(model_files, key=lambda x: x['modified'], reverse=True):
            size_mb = model['size'] / (1024 * 1024)
            print(f"    ✓ {model['name']} - 大小: {size_mb:.2f} MB - 修改时间: {model['modified']}")
    else:
        print("  未找到模型文件")
        print("  如需使用预训练模型，请下载或训练模型")
    
    return True

def check_data_directory():
    """检查数据目录"""
    print("\n" + "=" * 60)
    print("检查数据目录...")
    print("=" * 60)
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    if not os.path.exists(data_dir):
        print("  数据目录不存在")
        print("  如需训练模型，请先创建 data/ 目录并准备数据")
        return True
    
    # 检查BPH-PCA目录结构
    bph_pca_dir = os.path.join(data_dir, 'BPH-PCA')
    if os.path.exists(bph_pca_dir):
        print("  ✓ 找到 BPH-PCA 数据目录")
        
        # 检查子目录
        subdirs = ['BPH', 'PCA', 'ROI(BPH+PCA)']
        for subdir in subdirs:
            subdir_path = os.path.join(bph_pca_dir, subdir)
            if os.path.exists(subdir_path):
                print(f"    ✓ {subdir}/ 目录存在")
                
                # 检查模态目录
                if subdir in ['BPH', 'PCA']:
                    modalities = ['ADC', 'DWI', 'T2 fs', 'T2 not fs', 'gaoqing-T2']
                    for modality in modalities:
                        modality_path = os.path.join(subdir_path, modality)
                        if os.path.exists(modality_path):
                            n_files = len([f for f in os.listdir(modality_path) if f.endswith(('.nii', '.nii.gz', '.mha'))])
                            print(f"      ✓ {modality}/ - {n_files} 个文件")
                        else:
                            print(f"      - {modality}/ - 缺失")
            else:
                print(f"    - {subdir}/ - 缺失")
    else:
        print("  - BPH-PCA 数据目录不存在")
        print("  数据目录结构示例:")
        print("  data/")
        print("  └── BPH-PCA/")
        print("      ├── BPH/")
        print("      │   ├── ADC/")
        print("      │   ├── DWI/")
        print("      │   ├── T2 fs/")
        print("      │   ├── T2 not fs/")
        print("      │   └── gaoqing-T2/")
        print("      ├── PCA/")
        print("      │   └── ...")
        print("      └── ROI(BPH+PCA)/")
        print("          ├── BPH/")
        print("          └── PCA/")
    
    return True

def generate_report():
    """生成检查报告"""
    print("\n" + "=" * 60)
    print("生成检查报告...")
    print("=" * 60)
    
    report = {
        'check_time': datetime.now().isoformat(),
        'python_version': sys.version,
        'platform': sys.platform,
        'results': {}
    }
    
    # 执行各项检查
    file_ok = check_file_structure()
    deps_ok = check_dependencies()
    model_ok = check_model_files()
    data_ok = check_data_directory()
    
    report['results'] = {
        'file_structure': file_ok,
        'dependencies': deps_ok,
        'model_files': model_ok,
        'data_directory': data_ok
    }
    
    # 总体状态
    overall_ok = file_ok and deps_ok
    report['overall_status'] = overall_ok
    
    # 保存报告
    report_file = 'project_check_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n检查报告已保存到: {report_file}")
    
    # 输出总结
    print("\n" + "=" * 60)
    print("检查总结")
    print("=" * 60)
    
    if overall_ok:
        print("✓ 项目状态良好，可以正常运行")
        print("\n建议下一步:")
        if not os.path.exists('data/BPH-PCA'):
            print("  1. 准备数据并放置在 data/BPH-PCA/ 目录下")
            print("  2. 运行训练: python run.py train --config optimized")
        else:
            print("  1. 检查数据完整性")
            print("  2. 运行训练: python run.py train --config optimized")
    else:
        print("✗ 项目存在问题，需要修复")
        print("\n需要修复的问题:")
        if not file_ok:
            print("  - 缺失必需的项目文件")
        if not deps_ok:
            print("  - 缺失必需的依赖包")
        print("\n修复建议:")
        print("  1. 安装缺失的依赖: pip install -r requirements.txt")
        print("  2. 确保所有项目文件完整")
        print("  3. 重新运行检查: python check_project.py")
    
    return overall_ok

def main():
    """主函数"""
    print("前列腺多模态MRI分割项目 - 项目状态检查")
    print("=" * 60)
    
    # 生成检查报告
    success = generate_report()
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())