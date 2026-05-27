# 数据集处理模块
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import config

class FER2013Dataset(Dataset):
    """FER2013数据集类"""
    
    def __init__(self, csv_file, transform=None, augment=False):
        """
        Args:
            csv_file: CSV文件路径
            transform: 图像变换
            augment: 是否进行数据增强
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.augment = augment
        
        # 数据增强变换
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 读取像素数据
        pixels = self.data.iloc[idx]['pixels']
        emotion = self.data.iloc[idx]['emotion']
        
        # 将像素字符串转换为numpy数组
        pixels = np.array([int(p) for p in pixels.split()], dtype=np.uint8)
        image = pixels.reshape(config.IMAGE_SIZE, config.IMAGE_SIZE)
        
        # 转换为PIL图像
        image = Image.fromarray(image)
        
        # 数据增强
        if self.augment:
            image = self.augment_transform(image)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, emotion

def get_data_transforms():
    """获取数据变换"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return train_transform, test_transform

def get_data_loaders():
    """获取数据加载器"""
    train_transform, test_transform = get_data_transforms()
    
    # 创建数据集
    train_dataset = FER2013Dataset(
        csv_file=config.TRAIN_CSV,
        transform=train_transform,
        augment=True
    )
    
    test_dataset = FER2013Dataset(
        csv_file=config.TEST_CSV,
        transform=test_transform,
        augment=False
    )
    
    # 创建数据加载器（优化GPU训练）
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # 多线程加载数据
        pin_memory=True,  # 加速GPU数据传输
        persistent_workers=True  # 保持worker进程
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, test_loader
