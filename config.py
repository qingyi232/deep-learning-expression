# 配置文件
import os

# 数据集路径
DATA_DIR = 'FER'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')

# 模型参数
IMAGE_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 128  # 增大批大小以充分利用GPU
LEARNING_RATE = 0.0001  # 降低学习率避免NaN
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 15

# 表情类别
EMOTION_LABELS = {
    0: '生气 (Angry)',
    1: '厌恶 (Disgust)',
    2: '恐惧 (Fear)',
    3: '开心 (Happy)',
    4: '悲伤 (Sad)',
    5: '惊讶 (Surprise)',
    6: '中性 (Neutral)'
}

# 模型保存路径
MODEL_SAVE_PATH = 'models'
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'best_model.pth')

# OpenCV人脸检测器
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# 设备配置
DEVICE = 'cuda' if __name__ == '__main__' else 'cpu'
