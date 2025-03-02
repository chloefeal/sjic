from .base import BaseAlgorithm
from app.models import Algorithm, Alert
from app import db,app
import cv2
import numpy as np
import time
from datetime import datetime
from ultralytics.utils.files import increment_path
from pathlib import Path

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
            task_name = parameters.get('task_name')
            confidence = parameters.get('confidence', 0.5)
            algorithm_parameters = parameters.get('algorithm_parameters', {})
            on_alert = parameters.get('on_alert')  # 获取告警处理回调
            alertThreshold = parameters.get('alertThreshold', 5)
            last_alert_time = None
            camera_id = parameters.get('camera_id')
            
            # 获取检测区域
            detection_region = algorithm_parameters.get('detection_region')
            points = None
            frame_size = None
            roi_points = None
            if detection_region:
                points = detection_region.get('points', [])
                frame_size = detection_region.get('frame_size', {})

            frame_width, frame_height = int(camera.get(3)), int(camera.get(4))
            fps, fourcc = int(camera.get(5)), cv2.VideoWriter_fourcc(*"mp4v")
           
            ret, frame = camera.read()
            if not ret:
                return None
                
            if points and frame_size:
                # 计算实际图像和前端显示比例
                app.logger.debug(f"Original points: {points}")
                app.logger.debug(f"frame_size: {frame_size}")
                app.logger.debug(f"frame shape: {frame.shape}")

                scale_x = frame.shape[1] / frame_size['width']
                scale_y = frame.shape[0] / frame_size['height']

                app.logger.debug(f"scale_x: {scale_x}, scale_y: {scale_y}")
                
                # 转换点坐标 (左下角原点 -> 左上角原点) 并应用缩放
                roi_points = []
                for p in points:
                    x = int(p['x'] * scale_x)
                    # 将y坐标从左下角原点转换为左上角原点
                    y = int(frame.shape[0] - (p['y'] * scale_y))
                    roi_points.append((x, y))
                
                app.logger.debug(f"Converted roi_points: {roi_points}")

            while True:
                ret, frame = camera.read()
                if not ret:
                    time.sleep(1)
                    continue
                
                # 使用模型检测目标
                results = model(frame, conf=confidence)
                
                # 处理检测结果
                is_exception = False
                result_confidence = 0
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        app.logger.debug(f"Detection box: {box.xyxy[0]}")
                        box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        # foot_center = ((x1 + x2) // 2, y2)
                        app.logger.debug(f"Box center: {box_center}")

                        if roi_points:
                            # 检查点是否在检测区域内
                            if self.is_point_in_roi(box_center, roi_points):
                                is_exception = True
                                result_confidence = float(box.conf)
                                app.logger.debug(f"Point in ROI: True.")
                                app.logger.debug(f"result_confidence: {result_confidence}")
                                break

                #cv2.imshow('frame', results[0].plot())
                #cv2.waitKey(0)

                if is_exception and self.need_alert_again(last_alert_time, alertThreshold):
                    app.logger.debug(f"need_alert_again: True")
                    last_alert_time = datetime.now()
                    # TODO: 保存检测结果
                    save_filename = self.save_detection_image(frame, results)
                    ## save_dir = increment_path(Path("ultralytics_rc_output") / "exp", exist_ok=True, sep="", mkdir=True)
                    # save_dir = app.config['ALERT_FOLDER']
                    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    # save_filename = f'{task_name}_{timestamp}.mp4'
                    # video_writer = cv2.VideoWriter(str(save_dir / save_filename), fourcc, fps, (frame_width, frame_height))
                    app.logger.debug(f"save_filename: {save_filename}")
        
                    # 如果有检测结果，调用告警处理回调
                    if on_alert:
                        app.logger.debug(f"Calling on_alert")
                        on_alert(frame, {
                            'confidence': result_confidence,
                            'image_url': save_filename,
                        })
                               
        except Exception as e:
            app.logger.error(f"Error in object detection: {str(e)}")
            raise
