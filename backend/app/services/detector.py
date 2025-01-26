import cv2
import numpy as np
from ultralytics import YOLO
from app import socketio, db
import torch
from app.models import Alert, Camera, DetectionModel
import os

class DetectorService:
    def __init__(self):
        self.running = False
        self.camera = None
        self.model = None
        self.settings = None
        self.active_detectors = {}  # 存储每个摄像头的检测状态

    def load_camera(self, camera_id):
        """加载摄像头"""
        camera = Camera.query.get(camera_id)
        if not camera:
            raise ValueError(f"Camera with id {camera_id} not found")
        return camera
        
    def load_model(self, model_id):
        """加载YOLO模型"""
        try:
            model = DetectionModel.query.get(model_id)
            if not model:
                raise ValueError(f"Model with id {model_id} not found")

            model_path = os.path.join('models', model.path)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            return YOLO(model_path)
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

    def start(self, camera_id, model_id, settings):
        """启动检测"""
        try:
            # 检查是否已经在运行
            if camera_id in self.active_detectors:
                raise RuntimeError(f"Detection already running for camera {camera_id}")

            # 加载摄像头
            camera = self.load_camera(camera_id)
            rtsp_url = camera.get_rtsp_url()
            if not rtsp_url:
                raise ValueError(f"Invalid RTSP URL for camera {camera_id}")
            
            # 初始化视频捕获
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera stream: {rtsp_url}")
            
            # 加载模型
            model = self.load_model(model_id)

            # 保存检测器状态
            self.active_detectors[camera_id] = {
                'camera': cap,
                'model': model,
                'settings': settings,
                'running': True
            }
            
            # 开始检测循环
            self._start_detection_thread(camera_id)
            
        except Exception as e:
            if camera_id in self.active_detectors:
                self.stop(camera_id)
            raise RuntimeError(f"Failed to start detection: {str(e)}")
        
    def stop(self, camera_id):
        """停止检测"""
        if camera_id in self.active_detectors:
            detector = self.active_detectors[camera_id]
            detector['running'] = False
            if detector['camera']:
                detector['camera'].release()
            del self.active_detectors[camera_id]
            
    def _start_detection_thread(self, camera_id):
        """启动检测线程"""
        import threading
        thread = threading.Thread(
            target=self._detect_loop,
            args=(camera_id,),
            daemon=True
        )
        thread.start()
            
    def _detect_loop(self, camera_id):
        """检测循环"""
        detector = self.active_detectors[camera_id]
        
        while detector['running']:
            ret, frame = detector['camera'].read()
            if not ret:
                continue
                
            # 执行检测
            results = detector['model'](
                frame,
                conf=detector['settings'].get('confidence', 0.5)
            )
            
            # 处理检测结果
            for result in results:
                if result.conf >= detector['settings'].get('confidence', 0.5):
                    self.create_alert(
                        camera_id=camera_id,
                        alert_type=result.name,
                        confidence=float(result.conf),
                        image_url=self._save_detection_image(frame, result)
                    )
            
            # 发送结果到前端
            socketio.emit('detection_result', {
                'camera_id': camera_id,
                'frame': cv2.imencode('.jpg', frame)[1].tobytes()
            })

    def _save_detection_image(self, frame, result):
        """保存检测图片"""
        # TODO: 实现图片保存逻辑
        return None 