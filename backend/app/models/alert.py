from datetime import datetime, UTC
from app import db

class Alert(db.Model):
    __tablename__ = 'alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(UTC))
    camera_id = db.Column(db.Integer, db.ForeignKey('cameras.id'), nullable=False)
    alert_type = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float)
    image_url = db.Column(db.String(200))
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'camera_id': self.camera_id,
            'camera_name': self.camera.name,
            'alert_type': self.alert_type,
            'confidence': self.confidence,
            'image_url': self.image_url
        } 