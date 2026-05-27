# PyQt5图形用户界面
import sys
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QTextEdit, QTabWidget, QGroupBox, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont
import config
from predictor import EmotionPredictor

class VideoThread(QThread):
    """视频处理线程"""
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_result_signal = pyqtSignal(list)  # 新增：传递识别结果
    
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.running = False
        self.cap = None
    
    def run(self):
        """运行视频捕获"""
        self.cap = cv2.VideoCapture(0)
        self.running = True
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # 预测表情
                annotated_frame, results = self.predictor.predict_frame(frame)
                self.change_pixmap_signal.emit(annotated_frame)
                self.update_result_signal.emit(results)  # 发送识别结果
        
        self.cap.release()
    
    def stop(self):
        """停止视频捕获"""
        self.running = False
        self.wait()

class EmotionRecognitionGUI(QMainWindow):
    """表情识别系统主界面"""
    
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.video_thread = None
        self.init_ui()
        self.load_model()
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('基于深度学习的表情识别系统')
        self.setGeometry(100, 100, 1400, 900)  # 增大窗口尺寸
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 标题
        title_label = QLabel('基于深度学习的表情识别系统')
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont('Arial', 20, QFont.Bold)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)
        
        # 创建选项卡
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # 图像识别选项卡
        self.image_tab = self.create_image_tab()
        self.tabs.addTab(self.image_tab, '静态图像识别')
        
        # 实时识别选项卡
        self.video_tab = self.create_video_tab()
        self.tabs.addTab(self.video_tab, '实时视频识别')
        
        # 状态栏
        self.statusBar().showMessage('就绪')
    
    def create_image_tab(self):
        """创建图像识别选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # 图像显示区域
        image_group = QGroupBox('图像显示')
        image_layout = QVBoxLayout()
        image_group.setLayout(image_layout)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)  # 增大显示区域
        self.image_label.setStyleSheet('border: 2px solid #ccc; background-color: #f0f0f0;')
        self.image_label.setText('请上传图像')
        image_layout.addWidget(self.image_label)
        
        layout.addWidget(image_group)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.upload_btn = QPushButton('上传图像')
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setMinimumHeight(40)
        button_layout.addWidget(self.upload_btn)
        
        self.predict_btn = QPushButton('识别表情')
        self.predict_btn.clicked.connect(self.predict_image)
        self.predict_btn.setEnabled(False)
        self.predict_btn.setMinimumHeight(40)
        button_layout.addWidget(self.predict_btn)
        
        layout.addLayout(button_layout)
        
        # 结果显示区域
        result_group = QGroupBox('识别结果')
        result_layout = QVBoxLayout()
        result_group.setLayout(result_layout)
        
        self.image_result_text = QTextEdit()
        self.image_result_text.setReadOnly(True)
        self.image_result_text.setMaximumHeight(150)
        result_layout.addWidget(self.image_result_text)
        
        layout.addWidget(result_group)
        
        return tab
    
    def create_video_tab(self):
        """创建实时识别选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # 视频显示区域
        video_group = QGroupBox('视频显示')
        video_layout = QVBoxLayout()
        video_group.setLayout(video_layout)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)  # 增大显示区域
        self.video_label.setStyleSheet('border: 2px solid #ccc; background-color: #f0f0f0;')
        self.video_label.setText('点击"开始识别"启动摄像头')
        video_layout.addWidget(self.video_label)
        
        layout.addWidget(video_group)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.start_video_btn = QPushButton('开始识别')
        self.start_video_btn.clicked.connect(self.start_video)
        self.start_video_btn.setMinimumHeight(40)
        button_layout.addWidget(self.start_video_btn)
        
        self.stop_video_btn = QPushButton('停止识别')
        self.stop_video_btn.clicked.connect(self.stop_video)
        self.stop_video_btn.setEnabled(False)
        self.stop_video_btn.setMinimumHeight(40)
        button_layout.addWidget(self.stop_video_btn)
        
        layout.addLayout(button_layout)
        
        # 结果显示区域
        result_group = QGroupBox('实时识别结果')
        result_layout = QVBoxLayout()
        result_group.setLayout(result_layout)
        
        self.video_result_text = QTextEdit()
        self.video_result_text.setReadOnly(True)
        self.video_result_text.setMaximumHeight(100)
        result_layout.addWidget(self.video_result_text)
        
        layout.addWidget(result_group)
        
        return tab
    
    def load_model(self):
        """加载模型"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.predictor = EmotionPredictor(device=device)
            self.statusBar().showMessage(f'模型加载成功 (设备: {device})')
        except Exception as e:
            self.statusBar().showMessage(f'模型加载失败: {str(e)}')
            self.image_result_text.setText(f'错误: 模型加载失败\n{str(e)}\n\n请先训练模型！')
    
    def upload_image(self):
        """上传图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择图像', '', 
            'Image Files (*.png *.jpg *.jpeg *.bmp)'
        )
        
        if file_path:
            self.current_image_path = file_path
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.predict_btn.setEnabled(True)
            self.statusBar().showMessage(f'已加载图像: {file_path}')
    
    def predict_image(self):
        """预测图像表情"""
        if not hasattr(self, 'current_image_path'):
            return
        
        try:
            self.statusBar().showMessage('正在识别...')
            results, annotated_image = self.predictor.predict_image(self.current_image_path)
            
            # 显示标注后的图像
            height, width, channel = annotated_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(annotated_image.data, width, height, 
                           bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
            # 显示结果
            result_text = f'检测到 {len(results)} 个人脸\n\n'
            for i, result in enumerate(results):
                result_text += f'人脸 {i+1}:\n'
                result_text += f'  表情: {result["emotion"]}\n'
                result_text += f'  置信度: {result["confidence"]:.2%}\n'
                result_text += f'  概率分布:\n'
                for emotion, prob in result['probabilities'].items():
                    result_text += f'    {emotion}: {prob:.2%}\n'
                result_text += '\n'
            
            self.image_result_text.setText(result_text)
            self.statusBar().showMessage('识别完成')
            
        except Exception as e:
            self.image_result_text.setText(f'识别失败: {str(e)}')
            self.statusBar().showMessage('识别失败')
    
    def start_video(self):
        """开始视频识别"""
        if self.predictor is None:
            self.video_result_text.setText('错误: 模型未加载')
            return
        
        self.video_thread = VideoThread(self.predictor)
        self.video_thread.change_pixmap_signal.connect(self.update_video_frame)
        self.video_thread.update_result_signal.connect(self.update_video_result)  # 连接结果更新
        self.video_thread.start()
        
        self.start_video_btn.setEnabled(False)
        self.stop_video_btn.setEnabled(True)
        self.statusBar().showMessage('实时识别中...')
        self.statusBar().showMessage('实时识别中...')
    
    def stop_video(self):
        """停止视频识别"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        
        self.video_label.setText('点击"开始识别"启动摄像头')
        self.start_video_btn.setEnabled(True)
        self.stop_video_btn.setEnabled(False)
        self.statusBar().showMessage('已停止识别')
    
    def update_video_frame(self, frame):
        """更新视频帧"""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, 
                        bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    def update_video_result(self, results):
        """更新实时识别结果"""
        if len(results) == 0:
            self.video_result_text.setText('未检测到人脸')
            return
        
        result_text = f'检测到 {len(results)} 个人脸\n\n'
        for i, result in enumerate(results):
            emotion_cn = result["emotion"].split()[0]  # 中文部分
            result_text += f'人脸 {i+1}: {emotion_cn} ({result["confidence"]:.1%})\n'
        
        self.video_result_text.setText(result_text)
    
    def closeEvent(self, event):
        """关闭事件"""
        if self.video_thread:
            self.video_thread.stop()
        event.accept()

def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = EmotionRecognitionGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
