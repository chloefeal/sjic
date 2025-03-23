from .base import BaseAlgorithm
from app.models import Algorithm
from app import db, app
import cv2
import numpy as np
import time

class BeltBrokenSeries(BaseAlgorithm):
    """皮带表面故障检测算法-级联检测版本"""
    __tablename__ = 'algorithms'
    __mapper_args__ = {
        'polymorphic_identity': 'belt_broken_series'
    }

    @classmethod
    def register(cls):
        type_name = cls.__mapper_args__['polymorphic_identity']
        print(type_name)
        algorithm = Algorithm.query.filter_by(type=type_name).first()
        print(algorithm)
        algorithm = cls(
            name='皮带表面故障检测-高精度检测算法',
            type=type_name,
            description='皮带表面故障检测-高精度检测算法'
        )
        db.session.add(algorithm)
        db.session.commit()

    def process(self, camera, parameters):
        """处理图像"""
        try:
            # 获取YOLO模型
            yolo_model = parameters.get('model')
            # 获取Faster R-CNN模型
            rcnn_model = parameters.get('rcnn_model')
            
            confidence = parameters.get('confidence', 0.5)
            algorithm_parameters = parameters.get('algorithm_parameters', {})
            on_alert = parameters.get('on_alert')  # 获取告警处理回调
            
            # 从标定数据计算像素到厘米的转换比例
            calibration = algorithm_parameters.get('calibration', {})
            if calibration:
                belt_width = calibration.get('belt_width', 0)      # 真实宽度(cm)
                pixel_width = calibration.get('pixel_width', 0)    # 像素宽度
                frame_size = calibration.get('frame_size', {})     # 前端显示的帧尺寸
                points = calibration.get('points', [])             # 标定点
                
                # 获取实际图像尺寸
                ret, frame = camera.read()
                if ret:
                    actual_width = frame.shape[1]  # 实际图像宽度
                    scale_factor = actual_width / frame_size['width']  # 计算缩放比例
                    
                    # 计算实际的像素距离
                    actual_pixel_width = pixel_width * scale_factor
                    
                    # 计算真实的像素到厘米转换比例
                    pixel_to_cm = belt_width / actual_pixel_width
                else:
                    pixel_to_cm = 0.1  # 默认值
            else:
                pixel_to_cm = 0.1  # 默认值
            
            # 使用计算出的转换比例
            min_area_cm2 = algorithm_parameters.get('min_area_cm2', 100)
            
            while True:
                ret, frame = camera.read()
                if not ret:
                    continue

                # 使用YOLO Segment检测异常
                results = yolo_model(frame, conf=confidence)[0]
                
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
                        
                        # 计算原始YOLO检测的异常区域面积（像素）
                        yolo_area_px = np.sum(mask)
                        
                        # 使用的模型类型
                        model_used = 'yolo'
                        area_px = yolo_area_px

                        # 一般情况下只需要有一处破损，我们就需要报警
                        if conf < 0.5:
                            # 使用faster-rcnn进行再一次检测
                            if rcnn_model is not None:
                                try:
                                    # 扩大区域以包含更多上下文
                                    padding = 20
                                    roi_x1 = max(0, x1 - padding)
                                    roi_y1 = max(0, y1 - padding)
                                    roi_x2 = min(frame.shape[1], x2 + padding)
                                    roi_y2 = min(frame.shape[0], y2 + padding)
                                    
                                    # 提取感兴趣区域
                                    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                                    if roi.size == 0:
                                        continue
                                    
                                    # 使用 Faster R-CNN 检测
                                    rcnn_results = rcnn_model(roi)[0]
                                    
                                    # 处理 Faster R-CNN 检测结果
                                    if len(rcnn_results.boxes) > 0:
                                        # 获取最高置信度的检测结果
                                        best_conf = 0
                                        best_box = None
                                        
                                        for j in range(len(rcnn_results.boxes)):
                                            rcnn_box = rcnn_results.boxes[j]
                                            rcnn_conf = float(rcnn_box.conf[0])
                                            
                                            if rcnn_conf > best_conf:
                                                best_conf = rcnn_conf
                                                best_box = rcnn_box
                                        
                                        if best_box is not None and best_conf > 0.5:
                                            # 获取检测框坐标
                                            rx1, ry1, rx2, ry2 = map(int, best_box.xyxy[0])
                                            
                                            # 将坐标转换回原始图像坐标系
                                            rx1, ry1 = rx1 + roi_x1, ry1 + roi_y1
                                            rx2, ry2 = rx2 + roi_x1, ry2 + roi_y1
                                            
                                            # 更新检测框和置信度
                                            x1, y1, x2, y2 = rx1, ry1, rx2, ry2
                                            conf = best_conf
                                            model_used = 'rcnn'
                                            
                                            # 为RCNN检测结果创建新的掩码
                                            rcnn_mask = np.zeros_like(mask)
                                            # 将检测框区域设为1
                                            rcnn_mask[y1:y2, x1:x2] = 1
                                            
                                            # 计算RCNN检测的面积
                                            rcnn_area_px = np.sum(rcnn_mask)
                                            
                                            # 更新面积和掩码
                                            area_px = rcnn_area_px
                                            mask = rcnn_mask
                                            
                                            # 在原图上标记这是RCNN检测结果
                                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                                            cv2.putText(
                                                frame,
                                                f'RCNN: {conf:.2f}',
                                                (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5,
                                                (0, 165, 255),
                                                2
                                            )
                                except Exception as e:
                                    app.logger.error(f"Error in RCNN detection: {str(e)}")
                        else:
                            # 在原图上标记这是YOLO检测结果
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                frame,
                                f'YOLO: {conf:.2f}',
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2
                            )
                        
                        # 累加总面积
                        total_defect_area_px += area_px
                        
                        # 记录异常区域信息
                        defect_regions.append({
                            'x': x1,
                            'y': y1,
                            'width': x2 - x1,
                            'height': y2 - y1,
                            'area_px': area_px,
                            'confidence': conf,
                            'model': model_used
                        })
                        
                        # 在原图上绘制异常区域
                        colored_mask = np.zeros_like(frame)
                        colored_mask[mask == 1] = [0, 0, 255]  # 红色标记异常区域
                        frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)
                    
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
                            'confidence': max(r['confidence'] for r in defect_regions),
                            'models_used': list(set(r['model'] for r in defect_regions))
                        }
                
                # 如果没有检测到异常，返回原始帧
                return {
                    'frame': frame,
                    'alert': False
                }

        except Exception as e:
            app.logger.error(f"Error in belt broken algorithm: {str(e)}")
            raise
