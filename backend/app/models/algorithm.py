from app import db

class Algorithms(db.Model):
    __tablename__ = 'algorithms'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, default=0.5)
    alertThreshold = db.Column(db.Integer, default=5)  # 告警间隔（秒）
    regions = db.Column(db.JSON)  # 检测区域设置
    notificationEnabled = db.Column(db.Boolean, default=True)
    modelId = db.Column(db.Integer, db.ForeignKey('detection_models.id'))
    cameraId = db.Column(db.Integer, db.ForeignKey('cameras.id'))

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'confidence': self.confidence,
            'alertThreshold': self.alertThreshold,
            'regions': self.regions,
            'notificationEnabled': self.notificationEnabled,
            'modelId': self.modelId,
            'cameraId': self.cameraId
        } 
    