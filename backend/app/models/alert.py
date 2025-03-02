from datetime import datetime, timezone
from app import db

class Alert(db.Model):
    __tablename__ = 'alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    camera_id = db.Column(db.Integer, db.ForeignKey('cameras.id'), nullable=False)
    alert_type = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float)
    image_url = db.Column(db.String(200))
    
    def to_dict(self):
        # 构建完整的图片URL
        image_url = None
        if self.image_url:
            if self.image_url.startswith('http'):
                image_url = self.image_url
            else:
                # 使用相对路径
                image_url = f'/api/alerts/images/{self.image_url}'
        
        return {
            'id': self.id,
            'camera_id': self.camera_id,
            'camera_name': self.camera.name if self.camera else None,
            'alert_type': self.alert_type,
            'confidence': self.confidence,
            'image_url': image_url,
            'timestamp': self.timestamp.isoformat()
        } 