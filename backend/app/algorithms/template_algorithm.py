from .base import BaseAlgorithm
from app.models import Algorithm
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class TemplateAlgorithm(BaseAlgorithm):
    """模板匹配算法"""
    __tablename__ = 'algorithms'
    __mapper_args__ = {
        'polymorphic_identity': 'template_matching'
    }

    @classmethod
    def register(cls):
        type_name = cls.__mapper_args__['polymorphic_identity']
        algorithm = Algorithm.query.filter_by(type=type_name).first()
        print(algorithm)
        print(type_name)
        if not algorithm:
            algorithm = cls(
                name='模板匹配',
                type=type_name,
                description='使用OpenCV模板匹配算法，支持多模板匹配和相似度阈值设置',
                parameters=cls().get_parameters_schema()
            )
            db.session.add(algorithm)

    def process(self, frame, parameters):
        """模板匹配处理"""
        # 实现模板匹配逻辑
        pass

    def get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.8
                },
                "template_images": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "format": "uri"
                    }
                }
            }
        }

    def validate_parameters(self, parameters):
        return True 