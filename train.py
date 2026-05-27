# 模型训练模块
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import config
from model import get_model
from dataset import get_data_loaders

class Trainer:
    """模型训练器"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.model = get_model(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 训练历史
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.early_stop_counter = 0
        
        # 创建模型保存目录
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{running_loss/len(pbar):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs=config.NUM_EPOCHS):
        """完整训练流程"""
        print(f"开始训练，使用设备: {self.device}")
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # 学习率调整
            self.scheduler.step(val_loss)
            
            print(f"\n训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(config.BEST_MODEL_PATH)
                print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            # 早停
            if self.early_stop_counter >= config.EARLY_STOP_PATIENCE:
                print(f"\n早停触发，在epoch {epoch+1}停止训练")
                break
        
        print(f"\n训练完成！最佳验证准确率: {self.best_val_acc:.2f}%")
        self.plot_history()
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"模型加载成功，验证准确率: {self.best_val_acc:.2f}%")
    
    def plot_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True)
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='训练准确率')
        plt.plot(self.val_accs, label='验证准确率')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('训练和验证准确率')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.MODEL_SAVE_PATH, 'training_history.png'))
        print(f"训练历史图已保存到 {config.MODEL_SAVE_PATH}/training_history.png")

def main():
    """主训练函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 获取数据加载器
    print("加载数据集...")
    train_loader, test_loader = get_data_loaders()
    
    # 创建训练器
    trainer = Trainer(device=device)
    
    # 开始训练
    trainer.train(train_loader, test_loader)

if __name__ == '__main__':
    main()
