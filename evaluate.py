# 模型评估脚本
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import config
from model import get_model
from dataset import get_data_loaders

def evaluate_model():
    """评估模型性能"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("\n加载模型...")
    model = get_model(device)
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型加载成功，最佳验证准确率: {checkpoint.get('best_val_acc', 0):.2f}%")
    
    # 加载测试数据
    print("\n加载测试数据...")
    _, test_loader = get_data_loaders()
    
    # 预测
    print("\n开始评估...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='评估中'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算准确率
    accuracy = (all_preds == all_labels).mean() * 100
    print(f"\n测试集准确率: {accuracy:.2f}%")
    
    # 分类报告
    print("\n分类报告:")
    emotion_names = [config.EMOTION_LABELS[i].split()[0] for i in range(config.NUM_CLASSES)]
    print(classification_report(all_labels, all_preds, target_names=emotion_names))
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_names,
                yticklabels=emotion_names)
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    
    save_path = 'models/confusion_matrix.png'
    plt.savefig(save_path, dpi=150)
    print(f"\n混淆矩阵已保存到: {save_path}")
    
    # 每个类别的准确率
    print("\n各类别准确率:")
    for i in range(config.NUM_CLASSES):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_acc = (all_preds[class_mask] == all_labels[class_mask]).mean() * 100
            print(f"  {config.EMOTION_LABELS[i]}: {class_acc:.2f}%")

if __name__ == '__main__':
    evaluate_model()
