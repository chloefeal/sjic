from app import db
from datetime import datetime

class Task(db.Model):
    __tablename__ = 'tasks'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, default=0.5)
    alertThreshold = db.Column(db.Integer, default=5)  # 告警间隔（秒）
    regions = db.Column(db.JSON)  # 检测区域设置
    notificationEnabled = db.Column(db.Boolean, default=True)
    modelId = db.Column(db.Integer, db.ForeignKey('detection_models.id'))
    cameraId = db.Column(db.Integer, db.ForeignKey('cameras.id'))
    algorithm_id = db.Column(db.Integer, db.ForeignKey('algorithms.id'))
    status = db.Column(db.String(20), default='stopped')  # running, stopped, error
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'confidence': self.confidence,
            'alertThreshold': self.alertThreshold,
            'regions': self.regions,
            'notificationEnabled': self.notificationEnabled,
            'modelId': self.modelId,
            'cameraId': self.cameraId,
            'status': self.status,
            'created_at': self.created_at.isoformat()
        } 
    