from .base import BaseAlgorithm
from app.models import Algorithm
from app import db

class ObjectDetectionAlgorithm(BaseAlgorithm):
    """目标检测算法"""
    __tablename__ = 'algorithms'  # 使用同一个表
    __mapper_args__ = {
        'polymorphic_identity': 'object_detection'  # 改为和前端使用的类型值保持一致
    }

    @classmethod
    def register(cls):
        type_name = cls.__mapper_args__['polymorphic_identity']
        algorithm = Algorithm.query.filter_by(type=type_name).first()
        print(algorithm)
        print(type_name)
        if not algorithm:
            algorithm = cls(
                name='目标检测',
                type=type_name,
                description='使用YOLO模型进行目标检测，支持多目标检测和目标跟踪',
                parameters=cls().get_parameters_schema()
            )
            db.session.add(algorithm)
            db.session.commit()

    def process(self, camera, parameters):
        """目标检测算法"""
        while True:
            ret, frame = camera.read()
            if not ret:
                    continue
            model = parameters.get('model')
            confidence = parameters.get('confidence', 0.5)
            results = model(frame, conf=confidence)
            # 处理结果  
            return results

    def get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.5
                },
                "regions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "width": {"type": "number"},
                            "height": {"type": "number"}
                        }
                    }
                }
            }
        }

    def validate_parameters(self, parameters):
        # TODO: 实现参数验证
        return True 