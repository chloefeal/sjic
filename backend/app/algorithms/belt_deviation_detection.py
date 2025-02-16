from .base import BaseAlgorithm
from app import db
from app.models import Algorithm
import cv2
import numpy as np

class BeltDeviationDetection(BaseAlgorithm):
    """皮带跑偏检测算法"""
    __tablename__ = 'algorithms'
    __mapper_args__ = {
        'polymorphic_identity': 'belt_broken'
    }
    
    @classmethod
    def register(cls):
        type_name = cls.__mapper_args__['polymorphic_identity']
        print(type_name)
        algorithm = Algorithm.query.filter_by(type=type_name).first()
        print(algorithm)
        if not algorithm:
            algorithm = cls(
                name='皮带跑偏检测',
                type=type_name,
                description='皮带跑偏检测算法，当跑偏超过设置阈值时推送告警'
            )
            db.session.add(algorithm)
            db.session.commit()
    
    def process(self, camera, parameters):
        """处理图像"""
        try:
            model = parameters.get('model')
            confidence = parameters.get('confidence', 0.5)
            algorithm_parameters = parameters.get('algorithm_parameters', {})
            
            # 获取标定数据
            calibration = algorithm_parameters.get('calibration', {})
            if calibration:
                boundary_lines = calibration.get('boundary_lines', [])  # 边界线坐标
                frame_size = calibration.get('frame_size', {})         # 前端显示的帧尺寸
                boundary_distance = calibration.get('boundary_distance', 0)  # 边界线间实际距离(cm)
                deviation_threshold = calibration.get('deviation_threshold', 0)  # 跑偏报警阈值(cm)
                
                # 获取实际图像尺寸
                ret, frame = camera.read()
                if ret:
                    actual_width = frame.shape[1]  # 实际图像宽度
                    scale_factor = actual_width / frame_size['width']  # 计算缩放比例
                    
                    # 将前端坐标转换为实际图像坐标
                    actual_lines = []
                    for line in boundary_lines:
                        actual_line = []
                        for point in line:
                            x = int(point['x'] * scale_factor)
                            y = int(point['y'] * scale_factor)
                            actual_line.append((x, y))
                        actual_lines.append(actual_line)
                    
                    # 使用模型检测皮带边缘
                    results = model(frame, conf=confidence)
                    
                    # 计算皮带边缘到标定线的距离
                    # TODO: 根据实际检测结果计算距离并判断是否超出阈值
                    
                    # 返回处理结果
                    return {
                        'frame': frame,
                        'alert': False,  # 根据实际情况设置
                        'info': {
                            'deviation': 0,  # 实际偏移距离
                            'threshold': deviation_threshold
                        }
                    }
            
            return {
                'frame': frame,
                'alert': False
            }
                
        except Exception as e:
            app.logger.error(f"Error in belt deviation detection: {str(e)}")
            raise
