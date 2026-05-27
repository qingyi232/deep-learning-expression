# 数据预处理脚本
import pandas as pd
import numpy as np
import os
import config

def check_data():
    """检查数据集"""
    print("=" * 60)
    print("数据集检查")
    print("=" * 60)
    
    # 检查训练集
    if os.path.exists(config.TRAIN_CSV):
        train_df = pd.read_csv(config.TRAIN_CSV)
        print(f"\n训练集: {config.TRAIN_CSV}")
        print(f"  样本数量: {len(train_df)}")
        print(f"  列名: {list(train_df.columns)}")
        
        if 'emotion' in train_df.columns:
            print(f"\n  表情分布:")
            emotion_counts = train_df['emotion'].value_counts().sort_index()
            for emotion_id, count in emotion_counts.items():
                emotion_name = config.EMOTION_LABELS.get(emotion_id, f"未知({emotion_id})")
                print(f"    {emotion_name}: {count} ({count/len(train_df)*100:.2f}%)")
    else:
        print(f"\n训练集不存在: {config.TRAIN_CSV}")
    
    # 检查测试集
    if os.path.exists(config.TEST_CSV):
        test_df = pd.read_csv(config.TEST_CSV)
        print(f"\n测试集: {config.TEST_CSV}")
        print(f"  样本数量: {len(test_df)}")
        print(f"  列名: {list(test_df.columns)}")
        
        if 'emotion' in test_df.columns:
            print(f"\n  表情分布:")
            emotion_counts = test_df['emotion'].value_counts().sort_index()
            for emotion_id, count in emotion_counts.items():
                emotion_name = config.EMOTION_LABELS.get(emotion_id, f"未知({emotion_id})")
                print(f"    {emotion_name}: {count} ({count/len(test_df)*100:.2f}%)")
    else:
        print(f"\n测试集不存在: {config.TEST_CSV}")
    
    print("\n" + "=" * 60)

def process_fer2013():
    """处理FER2013原始数据集"""
    print("\n处理FER2013数据集...")
    
    # 检查是否存在icml_face_data.csv
    icml_path = os.path.join(config.DATA_DIR, 'icml_face_data.csv')
    
    if os.path.exists(icml_path):
        print(f"读取原始数据: {icml_path}")
        df = pd.read_csv(icml_path)
        
        # 清理列名（去除空格）
        df.columns = df.columns.str.strip()
        print(f"列名: {list(df.columns)}")
        
        # 分割训练集和测试集
        train_df = df[df['Usage'] == 'Training']
        test_df = df[df['Usage'] == 'PublicTest']
        
        # 只保留需要的列
        train_df = train_df[['emotion', 'pixels']]
        test_df = test_df[['emotion', 'pixels']]
        
        # 保存
        train_df.to_csv(config.TRAIN_CSV, index=False)
        test_df.to_csv(config.TEST_CSV, index=False)
        
        print(f"训练集保存到: {config.TRAIN_CSV} ({len(train_df)} 样本)")
        print(f"测试集保存到: {config.TEST_CSV} ({len(test_df)} 样本)")
    else:
        print(f"未找到原始数据文件: {icml_path}")
        print("请确保FER2013数据集已正确放置")

def main():
    """主函数"""
    # 检查数据
    check_data()
    
    # 如果训练集或测试集不存在，尝试处理原始数据
    if not os.path.exists(config.TRAIN_CSV) or not os.path.exists(config.TEST_CSV):
        process_fer2013()
        check_data()

if __name__ == '__main__':
    main()
