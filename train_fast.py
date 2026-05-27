# 快速GPU训练脚本（优化版）
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import config
from model import get_model
from dataset import get_data_loaders
import time

def train_fast():
    """快速GPU训练"""
    # 强制使用GPU
    if not torch.cuda.is_available():
        print("错误: 未检测到GPU，快速训练需要GPU支持")
        return
    
    device = torch.device('cuda')
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 启用cudnn自动优化
    torch.backends.cudnn.benchmark = True
    
    # 加载数据
    print("\n加载数据集...")
    train_loader, test_loader = get_data_loaders()
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    print(f"批大小: {config.BATCH_SIZE}")
    
    # 创建模型
    print("\n创建模型...")
    model = get_model(device)
    
    # 使用混合精度训练加速
    scaler = torch.cuda.amp.GradScaler()
    
    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练历史
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    early_stop_counter = 0
    
    # 创建保存目录
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    
    print(f"\n开始训练 (共{config.NUM_EPOCHS}个epoch)...")
    print("=" * 70)
    
    total_start_time = time.time()
    
    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]')
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # 更快的梯度清零
            
            # 混合精度训练
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]')
            for images, labels in pbar:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{running_loss/(pbar.n+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        val_loss = running_loss / len(test_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 学习率调整
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"\nEpoch {epoch+1} 完成 (耗时: {epoch_time:.1f}秒)")
        print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, config.BEST_MODEL_PATH)
            print(f"  ✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"  早停计数: {early_stop_counter}/{config.EARLY_STOP_PATIENCE}")
        
        # 早停
        if early_stop_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\n早停触发，停止训练")
            break
        
        print("-" * 70)
    
    total_time = time.time() - total_start_time
    print(f"\n训练完成！")
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    
    # 绘制训练历史
    plt.figure(figsize=(12, 4))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('训练和验证准确率')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODEL_SAVE_PATH, 'training_history.png'), dpi=150)
    print(f"\n训练历史图已保存")

if __name__ == '__main__':
    train_fast()
