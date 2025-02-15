from .base import BaseAlgorithm
from app.models import Algorithm
from app import db

class TemplateAlgorithm(BaseAlgorithm):
    """模板匹配算法"""
    __tablename__ = 'algorithms'
    __mapper_args__ = {
        'polymorphic_identity': 'template_matching'
    }

    @classmethod
    def register(cls):
        type_name = cls.__mapper_args__['polymorphic_identity']
        print(type_name)
        algorithm = Algorithm.query.filter_by(type=type_name).first()
        print(algorithm)
        if not algorithm:
            algorithm = cls(
                name='模板匹配',
                type=type_name,
                description='使用模板匹配进行目标检测'
            )
            db.session.add(algorithm)
            db.session.commit()

    def process(self, camera, parameters):
        """模板匹配处理"""
        # TODO: 实现模板匹配逻辑
        pass 