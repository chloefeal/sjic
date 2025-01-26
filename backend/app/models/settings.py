from app import db

class Settings(db.Model):
    __tablename__ = 'settings'
    
    id = db.Column(db.Integer, primary_key=True)
    confidence = db.Column(db.Float, default=0.5)
    alertThreshold = db.Column(db.Integer, default=5)  # 告警间隔（秒）
    regions = db.Column(db.JSON)  # 检测区域设置
    notificationEnabled = db.Column(db.Boolean, default=True)
    modelId = db.Column(db.Integer, db.ForeignKey('detection_models.id'))

    def to_dict(self):
        return {
            'id': self.id,
            'confidence': self.confidence_threshold,
            'alertThreshold': self.alert_interval,
            'regions': self.detection_regions,
            'notificationEnabled': self.notification_enabled,
            'modelId': self.detection_model_id
        } 
    