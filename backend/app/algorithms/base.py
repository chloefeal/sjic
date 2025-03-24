from abc import abstractmethod
from app import db, app
from app.models.algorithm import Algorithm
import requests
import os
from datetime import datetime
import cv2
from shapely.geometry import Point, Polygon

class BaseAlgorithm(Algorithm):
    """算法基类"""
    __abstract__ = True  # SQLAlchemy 不会为这个类创建表
    
    @abstractmethod
    def process(self, camera, parameters):
        """处理单帧图像"""
        pass

    @classmethod
    def get_subclasses(cls):
        """获取所有子类"""
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass

    @classmethod
    def register(cls):
        """注册算法到数据库"""
        pass

    @classmethod
    def register_algorithms(cls):
        """注册所有算法到数据库"""

        print("Registering algorithms...")
        print(f"Current file: {__file__}")
        for algorithm_class in cls.get_subclasses():
            print(f"Registering algorithm: {algorithm_class.__name__}")
            algorithm_class.register()
        db.session.commit()
        print("Algorithm registration completed") 

    def send_alert_to_external_api(self, alert_data):
        """发送告警到外部 API"""
        try:
            # 配置外部 API 的 URL
            api_url = app.config.get('EXTERNAL_ALERT_API_URL')
            if not api_url:
                app.logger.warning("External alert API URL not configured")
                return

            # 发送 POST 请求
            response = requests.post(
                api_url,
                json=alert_data,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f"Bearer {app.config.get('EXTERNAL_API_TOKEN')}"
                }
            )
            
            if not response.ok:
                app.logger.error(f"Failed to send alert to external API: {response.text}")
                
        except Exception as e:
            app.logger.error(f"Error sending alert to external API: {str(e)}")

    # return filename, None
    def save_detection_image(self, frame, results, task_name):
        """保存检测图片"""
        try:
            # 创建保存目录
            save_dir = app.config['ALERT_FOLDER']
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'{task_name}_{timestamp}.jpg'
            filepath = os.path.join(save_dir, filename)
            
            # 保存图像
            cv2.imwrite(filepath, frame)
            
            return filename
            
        except Exception as e:
            app.logger.error(f"Error saving detection image: {str(e)}")
            return None 
        
    def need_alert_again(self, last_alert_time, alertThreshold):
        """判断是否需要再次告警"""
        if last_alert_time is None:
            return True
        return (datetime.now() - last_alert_time).total_seconds() >= alertThreshold

    def is_point_in_roi(self, point, points):
        """
        判断点是否在检测区域内
        :param point: 点 (x, y)
        :param points: 检测区域的边界点列表 [(x1, y1), (x2, y2), ..., (xn, yn)]
        :return: 如果点在检测区域内返回True，否则返回False
        """
        # 创建一个Point对象
        app.logger.debug(f"is_point_in_roi:point: {point}")
        center_point = Point(point)
        # 创建一个Polygon对象
        roi_polygon = Polygon(points)
        # 使用contains方法判断点是否在多边形内
        return roi_polygon.contains(center_point)
    


