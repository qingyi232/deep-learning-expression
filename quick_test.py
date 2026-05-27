# 快速测试脚本
import torch
import sys

def test_environment():
    """测试环境配置"""
    print("=" * 60)
    print("环境测试")
    print("=" * 60)
    
    # Python版本
    print(f"\nPython版本: {sys.version}")
    
    # PyTorch
    print(f"\nPyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 导入测试
    print("\n模块导入测试:")
    try:
        import cv2
        print(f"  ✓ OpenCV: {cv2.__version__}")
    except:
        print("  ✗ OpenCV导入失败")
    
    try:
        from PyQt5.QtWidgets import QApplication
        print(f"  ✓ PyQt5")
    except:
        print("  ✗ PyQt5导入失败")
    
    try:
        import pandas as pd
        print(f"  ✓ Pandas: {pd.__version__}")
    except:
        print("  ✗ Pandas导入失败")
    
    try:
        import numpy as np
        print(f"  ✓ NumPy: {np.__version__}")
    except:
        print("  ✗ NumPy导入失败")
    
    try:
        import matplotlib
        print(f"  ✓ Matplotlib: {matplotlib.__version__}")
    except:
        print("  ✗ Matplotlib导入失败")
    
    print("\n" + "=" * 60)

def test_data():
    """测试数据集"""
    print("\n数据集测试")
    print("=" * 60)
    
    import os
    import pandas as pd
    import config
    
    if os.path.exists(config.TRAIN_CSV):
        train_df = pd.read_csv(config.TRAIN_CSV)
        print(f"\n✓ 训练集: {len(train_df)} 样本")
    else:
        print(f"\n✗ 训练集不存在: {config.TRAIN_CSV}")
    
    if os.path.exists(config.TEST_CSV):
        test_df = pd.read_csv(config.TEST_CSV)
        print(f"✓ 测试集: {len(test_df)} 样本")
    else:
        print(f"✗ 测试集不存在: {config.TEST_CSV}")
    
    print("\n" + "=" * 60)

def test_model():
    """测试模型"""
    print("\n模型测试")
    print("=" * 60)
    
    import os
    import config
    from model import get_model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = get_model(device)
        print(f"\n✓ 模型创建成功")
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        # 测试前向传播
        dummy_input = torch.randn(1, 1, 48, 48).to(device)
        output = model(dummy_input)
        print(f"  输出形状: {output.shape}")
        print(f"✓ 模型前向传播测试通过")
        
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
    
    # 检查模型文件
    if os.path.exists(config.BEST_MODEL_PATH):
        print(f"\n✓ 训练好的模型存在: {config.BEST_MODEL_PATH}")
    else:
        print(f"\n✗ 训练好的模型不存在，需要先训练模型")
    
    print("\n" + "=" * 60)

def main():
    """主函数"""
    test_environment()
    test_data()
    test_model()
    
    print("\n测试完成！")
    print("\n下一步:")
    print("  1. 如果模型不存在，运行: python train_fast.py")
    print("  2. 训练完成后，运行: python gui.py")

if __name__ == '__main__':
    main()
