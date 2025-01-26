import cv2
import numpy as np
from ultralytics import YOLO
from app import socketio, db
import torch
from app.models import Alert

class DetectorService:
    def __init__(self):
        self.models = {}
        
    def load_model(self, model_path):
        """加载YOLO模型"""
        try:
            model = YOLO(model_path)
            return model
        except Exception as e:
            raise Exception(f"模型加载失败: {str(e)}")
            
    def create_alert(self, camera_id, alert_type, confidence, image_url):
        """创建告警记录"""
        alert = Alert(
            camera_id=camera_id,
            alert_type=alert_type,
            confidence=confidence,
            image_url=image_url
        )
        
        db.session.add(alert)
        db.session.commit()
        
        # 发送实时告警
        socketio.emit('new_alert', alert.to_dict())
        
        return alert
        
    def detect(self, image, model_name, conf_threshold=0.5):
        """执行目标检测"""
        if model_name not in self.models:
            raise Exception("模型未加载")
            
        results = self.models[model_name](image, conf=conf_threshold)
        
        # 如果检测到目标，创建告警
        if len(results) > 0:
            for result in results:
                if result.conf >= conf_threshold:
                    self.create_alert(
                        camera_id=self.camera_id,
                        alert_type=result.name,
                        confidence=float(result.conf),
                        image_url=self.save_detection_image(image, result)
                    )
                    
        return results

class Detector:
    def __init__(self):
        self.running = False
        self.camera = None
        self.model = None
        self.settings = None
        
    def start(self, camera_id, model_type, settings):
        self.camera = cv2.VideoCapture(camera_id)
        self.model = YOLO(f'models/{model_type}.pt')
        self.settings = settings
        self.running = True
        self._detect_loop()
        
    def stop(self):
        self.running = False
        if self.camera:
            self.camera.release()
            
    def _detect_loop(self):
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                continue
                
            # 执行检测
            results = self.model(frame)
            
            # 处理检测结果
            alerts = self._process_results(results)
            
            # 发送结果到前端
            socketio.emit('detection_result', {
                'alerts': alerts,
                'frame': cv2.imencode('.jpg', frame)[1].tobytes()
            }) 