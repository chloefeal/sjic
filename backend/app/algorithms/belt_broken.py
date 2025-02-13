from .base import BaseAlgorithm
from app.models import Algorithm
from app import db
import cv2
import numpy as np

class BeltBroken(BaseAlgorithm):
    """皮带表面故障检测算法"""
    __tablename__ = 'algorithms'
    __mapper_args__ = {
        'polymorphic_identity': 'belt_broken'
    }

    @classmethod
    def register(cls):
        type_name = cls.__mapper_args__['polymorphic_identity']
        algorithm = Algorithm.query.filter_by(type=type_name).first()
        if not algorithm:
            algorithm = cls(
                name='皮带表面故障检测',
                type=type_name,
                description='使用Segment检测皮带表面异常，当异常面积大于100cm²时推送告警'
            )
            db.session.add(algorithm)
            db.session.commit()

    def process(self, camera, parameters):
        """皮带表面故障检测处理"""
        model = parameters.get('model')  # 应该是YOLO Segment模型
        confidence = parameters.get('confidence', 0.5)
        pixel_to_cm = parameters.get('pixel_to_cm', 0.1)  # 像素到厘米的转换比例
        min_area_cm2 = parameters.get('min_area_cm2', 100)  # 最小异常面积(cm²)
        
        while True:
            ret, frame = camera.read()
            if not ret:
                continue

            # 使用YOLO Segment检测异常
            results = model(frame, conf=confidence)[0]
            
            if len(results) > 0:
                defect_regions = []
                total_defect_area_px = 0
                
                # 处理每个检测到的异常区域
                for i in range(len(results.boxes)):
                    # 获取检测框
                    box = results.boxes[i]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # 获取分割掩码
                    mask = results.masks[i].data[0].cpu().numpy()
                    
                    # 计算异常区域面积（像素）
                    area_px = np.sum(mask)
                    total_defect_area_px += area_px
                    
                    # 记录异常区域信息
                    defect_regions.append({
                        'x': x1,
                        'y': y1,
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'area_px': area_px,
                        'confidence': conf
                    })
                    
                    # 在原图上绘制异常区域
                    colored_mask = np.zeros_like(frame)
                    colored_mask[mask == 1] = [0, 0, 255]  # 红色标记异常区域
                    frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)
                    
                    # 绘制边界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 计算总异常面积(cm²)
                total_defect_area_cm2 = total_defect_area_px * (pixel_to_cm ** 2)
                
                # 如果异常面积超过阈值，返回结果
                if total_defect_area_cm2 >= min_area_cm2:
                    # 添加文本说明
                    text = f'Defect Area: {total_defect_area_cm2:.1f} cm²'
                    cv2.putText(
                        frame,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                    
                    return {
                        'frame': frame,
                        'defect_area': total_defect_area_cm2,
                        'defect_regions': defect_regions,
                        'alert': True,
                        'alert_type': 'belt_broken',
                        'confidence': max(r['confidence'] for r in defect_regions)
                    }
            
            # 如果没有检测到异常，返回原始帧
            return {
                'frame': frame,
                'alert': False
            }
