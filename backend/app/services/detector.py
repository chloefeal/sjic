import cv2
import numpy as np
from ultralytics import YOLO
from app import socketio, db, app
import torch
from app.models import Alert, Camera, DetectionModel, Task
import os
from app.models.algorithm import Algorithm
from config import Config
import logging

app.logger = logging.getLogger(__name__)

class DetectorService:
    def __init__(self):
        self.running = False
        self.camera = None
        self.model = None
        self.task = None
        self.active_detectors = {}  # 存储每个任务的检测状态

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

            # 使用绝对路径
            model_path = os.path.join(Config.MODEL_FOLDER, model.path)
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

    def start(self, camera_id, model_id, task_id):
        """启动检测"""
        try:
            app.logger.info(f"Starting detection: camera={camera_id}, model={model_id}, task={task_id}")
            
            # 检查是否已经在运行
            if task_id in self.active_detectors:
                msg = f"Detection already running for camera {camera_id}"
                app.logger.warning(msg)
                raise RuntimeError(msg)

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
            self.active_detectors[task_id] = {
                'camera': cap,
                'model': model,
                'task_id': task_id,
                'running': True
            }
            
            # 开始检测循环
            self._start_detection_thread(task_id)
            
            app.logger.info(f"Detection started successfully for task {task_id}")
            
        except Exception as e:
            app.logger.error(f"Failed to start detection: {str(e)}", exc_info=True)
            if task_id in self.active_detectors:
                self.stop(task_id)
            raise RuntimeError(f"Failed to start detection: {str(e)}")
        
    def stop(self, task_id):
        """停止检测"""
        if task_id in self.active_detectors:
            detector = self.active_detectors[task_id]
            detector['running'] = False
            if detector['camera']:
                detector['camera'].release()
            del self.active_detectors[task_id]
            
    def _start_detection_thread(self, task_id):
        """启动检测线程"""
        import threading
        thread = threading.Thread(
            target=self._detect_loop,
            args=(task_id,),
            daemon=True
        )
        thread.start()
            
    def _detect_loop(self, task_id):
        """检测循环"""
        with app.app_context():  # 添加应用上下文
            try:
                detector = self.active_detectors[task_id]
                task = Task.query.get(detector['task_id'])
                algorithm = Algorithm.get_algorithm_by_id(task.algorithm_id)
                
                while detector['running']:
                    # 使用选定的算法处理帧
                    results = algorithm.process(detector['camera'], {
                        'model': detector['model'],
                        'confidence': task.confidence,
                        'alertThreshold': task.alertThreshold,
                        'algorithm_parameters': task.algorithm_parameters
                    })
                    
                    # 处理结果并创建告警
                    self._handle_results(task_id, results)
                    frame = results['frame']
                    
                    # 发送结果到前端
                    socketio.emit('detection_result', {
                        'task_id': task_id,
                        'frame': cv2.imencode('.jpg', frame)[1].tobytes()
                    })
            except Exception as e:
                app.logger.error(f"Error in detection loop: {str(e)}")
                self.stop(task_id)

    def _handle_results(self, task_id, results):
        """处理检测结果"""
        with app.app_context():  # 添加应用上下文
            try:
                if results.get('alert'):
                    detector = self.active_detectors[task_id]
                    task = Task.query.get(detector['task_id'])
                    
                    # 创建告警记录
                    alert = Alert(
                        task_id=task_id,
                        camera_id=task.cameraId,
                        alert_type=task.algorithm_type,
                        confidence=results.get('confidence', 0),
                        image_path=self._save_detection_image(results['frame'], results)
                    )
                    db.session.add(alert)
                    db.session.commit()
                    
                    # 发送告警到前端
                    socketio.emit('new_alert', alert.to_dict())
            except Exception as e:
                app.logger.error(f"Error handling results: {str(e)}")

    def _save_detection_image(self, frame, result):
        """保存检测图片"""
        # TODO: 实现图片保存逻辑
        return None 