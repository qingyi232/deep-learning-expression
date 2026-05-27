# 主程序入口
import sys
import argparse
import torch

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于深度学习的表情识别系统')
    parser.add_argument('--mode', type=str, default='gui', 
                       choices=['train', 'gui', 'check'],
                       help='运行模式: train(训练), gui(图形界面), check(检查数据)')
    parser.add_argument('--gpu', action='store_true', 
                       help='使用GPU训练')
    
    args = parser.parse_args()
    
    if args.mode == 'check':
        # 检查数据集
        from preprocess_data import main as check_main
        check_main()
    
    elif args.mode == 'train':
        # 训练模型
        print("=" * 60)
        print("开始训练模型")
        print("=" * 60)
        
        # 检查GPU
        if torch.cuda.is_available():
            print(f"GPU可用: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
        else:
            print("警告: GPU不可用，将使用CPU训练（速度较慢）")
        
        from train import main as train_main
        train_main()
    
    elif args.mode == 'gui':
        # 启动图形界面
        from gui import main as gui_main
        gui_main()

if __name__ == '__main__':
    main()
