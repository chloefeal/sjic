from app import db
from datetime import datetime

class Task(db.Model):
    __tablename__ = 'tasks'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, default=0.5)  # 通用参数
    alertThreshold = db.Column(db.Integer, default=5)  # 通用参数
    notificationEnabled = db.Column(db.Boolean, default=True)  # 通用参数
    modelId = db.Column(db.Integer, db.ForeignKey('detection_models.id'))
    cameraId = db.Column(db.Integer, db.ForeignKey('cameras.id'))
    algorithm_id = db.Column(db.Integer, db.ForeignKey('algorithms.id'))
    algorithm_parameters = db.Column(db.JSON)  # 所有算法特定参数
    status = db.Column(db.String(20), default='stopped')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'confidence': self.confidence,
            'alertThreshold': self.alertThreshold,
            'notificationEnabled': self.notificationEnabled,
            'modelId': self.modelId,
            'cameraId': self.cameraId,
            'algorithm_id': self.algorithm_id,
            'algorithm_parameters': self.algorithm_parameters,
            'status': self.status,
            'created_at': self.created_at.isoformat()
        } 
    