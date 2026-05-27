# 表情预测模块
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import config
from model import get_model

class EmotionPredictor:
    """表情识别预测器"""
    
    def __init__(self, model_path=config.BEST_MODEL_PATH, device='cpu'):
        """
        Args:
            model_path: 模型权重路径
            device: 计算设备
        """
        self.device = device
        self.model = get_model(device)
        
        # 加载模型权重
        try:
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"模型加载成功: {model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # 加载人脸检测器
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_faces(self, image):
        """
        检测图像中的人脸
        
        Args:
            image: OpenCV格式的图像 (BGR)
        
        Returns:
            faces: 人脸区域列表 [(x, y, w, h), ...]
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 尝试多种参数组合检测人脸
        # 参数1: 标准参数
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # 如果没检测到，使用更宽松的参数
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20)
            )
        
        # 如果还是没检测到，使用最宽松的参数
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.02,
                minNeighbors=2,
                minSize=(10, 10)
            )
        
        return faces
    
    def predict_emotion(self, face_image):
        """
        预测单个人脸的表情
        
        Args:
            face_image: PIL Image或numpy数组
        
        Returns:
            emotion_label: 表情标签
            confidence: 置信度
            probabilities: 所有类别的概率分布
        """
        # 转换为PIL Image
        if isinstance(face_image, np.ndarray):
            face_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        
        # 预处理
        image_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        emotion_id = predicted.item()
        emotion_label = config.EMOTION_LABELS[emotion_id]
        confidence_value = confidence.item()
        prob_dict = {config.EMOTION_LABELS[i]: probabilities[0][i].item() 
                     for i in range(config.NUM_CLASSES)}
        
        return emotion_label, confidence_value, prob_dict
    
    def predict_image(self, image_path):
        """
        预测图像文件中的表情
        
        Args:
            image_path: 图像文件路径
        
        Returns:
            results: 预测结果列表
            annotated_image: 标注后的图像
        """
        # 读取图像 - 使用支持中文路径的方法
        try:
            # 方法1: 使用numpy读取，支持中文路径
            import numpy as np
            with open(image_path, 'rb') as f:
                image_data = np.frombuffer(f.read(), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
        except Exception as e:
            raise ValueError(f"图像读取失败: {str(e)}")
        
        # 检测人脸
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            raise ValueError("未检测到人脸，请使用包含清晰人脸的图片")
        
        results = []
        annotated_image = image.copy()
        
        for (x, y, w, h) in faces:
            # 提取人脸区域
            face_roi = image[y:y+h, x:x+w]
            
            # 预测表情
            emotion_label, confidence, probabilities = self.predict_emotion(face_roi)
            
            results.append({
                'bbox': (x, y, w, h),
                'emotion': emotion_label,
                'confidence': confidence,
                'probabilities': probabilities
            })
            
            # 在图像上标注
            # 绘制人脸框（绿色，加粗）
            cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # 准备文字标签
            emotion_text = emotion_label.split()[0]  # 只取中文部分
            text = f"{emotion_text} {confidence:.1%}"
            
            # 计算文字大小
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # 在人脸框下方绘制文字背景（深绿色半透明）
            text_x = x
            text_y = y + h + text_height + 10
            
            # 如果文字会超出图像底部，则放在人脸框上方
            if text_y > annotated_image.shape[0] - 5:
                text_y = y - 10
            
            # 绘制文字背景矩形
            cv2.rectangle(annotated_image, 
                         (text_x, text_y - text_height - 5), 
                         (text_x + text_width + 10, text_y + baseline + 5),
                         (0, 180, 0), -1)
            
            # 绘制白色文字
            cv2.putText(annotated_image, text, (text_x + 5, text_y),
                       font, font_scale, (255, 255, 255), thickness)
        
        return results, annotated_image
    
    def predict_frame(self, frame):
        """
        预测视频帧中的表情（用于实时识别）
        
        Args:
            frame: OpenCV格式的视频帧
        
        Returns:
            annotated_frame: 标注后的帧
            results: 预测结果列表
        """
        # 检测人脸
        faces = self.detect_faces(frame)
        
        results = []
        annotated_frame = frame.copy()
        
        for (x, y, w, h) in faces:
            # 提取人脸区域
            face_roi = frame[y:y+h, x:x+w]
            
            # 预测表情
            emotion_label, confidence, probabilities = self.predict_emotion(face_roi)
            
            results.append({
                'bbox': (x, y, w, h),
                'emotion': emotion_label,
                'confidence': confidence,
                'probabilities': probabilities
            })
            
            # 在图像上标注
            # 绘制人脸框（绿色，加粗）
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # 准备文字标签 - 只使用英文和数字，避免中文乱码
            emotion_dict = {
                '生气': 'Angry',
                '厌恶': 'Disgust', 
                '恐惧': 'Fear',
                '开心': 'Happy',
                '悲伤': 'Sad',
                '惊讶': 'Surprise',
                '中性': 'Neutral'
            }
            
            emotion_cn = emotion_label.split()[0]  # 中文部分
            emotion_en = emotion_dict.get(emotion_cn, emotion_label.split()[-1].strip('()'))  # 英文部分
            text = f"{emotion_en} {confidence:.1%}"
            
            # 计算文字大小
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # 在人脸框下方绘制文字背景（深绿色）
            text_x = x
            text_y = y + h + text_height + 10
            
            # 如果文字会超出图像底部，则放在人脸框上方
            if text_y > annotated_frame.shape[0] - 5:
                text_y = y - 10
            
            # 绘制文字背景矩形
            cv2.rectangle(annotated_frame, 
                         (text_x, text_y - text_height - 5), 
                         (text_x + text_width + 10, text_y + baseline + 5),
                         (0, 180, 0), -1)
            
            # 绘制白色文字
            cv2.putText(annotated_frame, text, (text_x + 5, text_y),
                       font, font_scale, (255, 255, 255), thickness)
        
        return annotated_frame, results
