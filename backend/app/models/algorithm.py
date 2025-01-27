from app import db
from datetime import datetime

class Algorithm(db.Model):
    __tablename__ = 'algorithms'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(50), nullable=False)  # 这个列用来区分不同的算法类型
    description = db.Column(db.Text)
    parameters = db.Column(db.JSON)  # 算法参数配置
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __mapper_args__ = {
        'polymorphic_identity': 'algorithm',
        'polymorphic_on': type  # 指定使用 type 列来区分不同的算法
    }
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'description': self.description,
            'parameters': self.parameters,
            'created_at': self.created_at.isoformat()
        }

    @classmethod
    def get_algorithm(cls, algorithm_type):
        """获取算法实例"""
        algorithm = cls.query.filter_by(type=algorithm_type).first()
        if not algorithm:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
        return algorithm 