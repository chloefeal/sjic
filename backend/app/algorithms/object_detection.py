from .base import BaseAlgorithm
from app.models import Algorithm
from app import db,app
import cv2
import numpy as np
import time

class ObjectDetectionAlgorithm(BaseAlgorithm):
    """目标检测算法"""
    __tablename__ = 'algorithms'
    __mapper_args__ = {
        'polymorphic_identity': 'object_detection'
    }

    @classmethod
    def register(cls):
        type_name = cls.__mapper_args__['polymorphic_identity']
        algorithm = Algorithm.query.filter_by(type=type_name).first()
        if not algorithm:
            algorithm = cls(
                name='目标检测',
                type=type_name,
                description='目标检测，支持多目标检测和目标跟踪，可指定检测区域'
            )
            db.session.add(algorithm)
            db.session.commit()

    def process(self, camera, parameters):
        """目标检测算法"""
        try:
            model = parameters.get('model')
            confidence = parameters.get('confidence', 0.5)
            algorithm_parameters = parameters.get('algorithm_parameters', {})
            on_alert = parameters.get('on_alert')  # 获取告警处理回调
            
            # 获取检测区域
            detection_region = algorithm_parameters.get('detection_region')
            points = None
            frame_size = None
            if detection_region:
                points = detection_region.get('points', [])
                frame_size = detection_region.get('frame_size', {})
            
            ret, frame = camera.read()
            if not ret:
                return None
                
            if points and frame_size:
                # 计算实际图像和前端显示比例
                app.logger.debug(f"points: {points}")
                app.logger.debug(f"frame_size: {frame_size}")
                app.logger.debug(f"frame: {frame.shape}")

                scale_x = frame.shape[1] / frame_size['width']
                scale_y = frame.shape[0] / frame_size['height']

                app.logger.debug(f"scale_x: {scale_x}")
                app.logger.debug(f"scale_y: {scale_y}")
                
                # 转换点坐标
                roi_points = np.array([
                    [int(p['x'] * scale_x), int(p['y'] * scale_y)]
                    for p in points
                ], np.int32)

                app.logger.debug(f"roi_points: {roi_points}")
            
            while True:
                ret, frame = camera.read()
                if not ret:
                    time.sleep(1)
                    continue
                # 使用模型检测目标
                results = model(frame, conf=confidence)
                
                # 处理检测结果
                detections = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        app.logger.debug(f"box: {box.xyxy[0]}")
                        box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        
                        detections.append({
                            'box': box.xyxy[0].tolist(),
                            'confidence': float(box.conf),
                            'class': int(box.cls)
                        })
            
                # 如果有检测结果，调用告警处理回调
                if detections and on_alert:
                    on_alert(frame, {
                        'detections': detections,
                        'confidence': max(d['confidence'] for d in detections),
                        'alert': True
                    })
                               
        except Exception as e:
            app.logger.error(f"Error in object detection: {str(e)}")
            raise
