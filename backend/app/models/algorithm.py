from app import db
from datetime import datetime

class Algorithm(db.Model):
    __tablename__ = 'algorithms'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(50), nullable=False)  # 算法类型：如 'object_detection', 'behavior_analysis' 等
    description = db.Column(db.Text)
    parameters = db.Column(db.JSON)  # 算法参数配置
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'description': self.description,
            'parameters': self.parameters,
            'created_at': self.created_at.isoformat()
        } 