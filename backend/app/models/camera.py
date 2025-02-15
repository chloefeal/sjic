from datetime import datetime
from app import db
from werkzeug.utils import secure_filename
from config import Config
import os

class Camera(db.Model):
    __tablename__ = 'cameras'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    url = db.Column(db.String(200), nullable=False)
    status = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    alerts = db.relationship('Alert', backref='camera', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'url': self.url,
            'status': self.status,
            'created_at': self.created_at.isoformat()
        }

    def get_rtsp_url(self):
        file = self.url
        if file.startswith('rtsp://'):
            return self.url
        else:
            filename = secure_filename(file.filename)
            file_path = os.path.join(Config.VIDEO_FOLDER, filename)
            return file_path
