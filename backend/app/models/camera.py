from datetime import datetime
from flask import current_app
from app.extensions import db
from werkzeug.utils import secure_filename
from config import Config
import os

# 摄像头相对考生的安装位置；姿态算法按此切换几何判断。
MOUNT_POSITIONS = ('front_top', 'back_top', 'side_top', 'top')
MOUNT_POSITION_LABELS = {
    'front_top': '前上方',
    'back_top': '后上方',
    'side_top': '侧上方',
    'top': '正上方',
}
DEFAULT_MOUNT_POSITION = 'back_top'


def normalize_mount_position(value):
    v = (value or '').strip()
    return v if v in MOUNT_POSITIONS else DEFAULT_MOUNT_POSITION


class Camera(db.Model):
    __tablename__ = 'cameras'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)  # 机位用名称体现，如「1号考生位」
    url = db.Column(db.String(200), nullable=False)
    # front_top / back_top / side_top / top；科目一考场常见后上方
    mount_position = db.Column(db.String(20), nullable=False, default=DEFAULT_MOUNT_POSITION)
    status = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    alerts = db.relationship('Alert', backref='camera', lazy=True)
    
    def to_dict(self):
        pos = normalize_mount_position(self.mount_position)
        return {
            'id': self.id,
            'name': self.name,
            'url': self.url,
            'mount_position': pos,
            'mount_position_label': MOUNT_POSITION_LABELS.get(pos, pos),
            'status': self.status,
            'created_at': self.created_at.isoformat()
        }

    def get_rtsp_url(self):
        file = str(self.url).strip()
        current_app.logger.info(f"Camera Source: {file}")
        
        if any(file.startswith(p) for p in ['rtsp://', 'rtmp://', 'http://', 'https://']):
            return file
            
        if file.isdigit():
            return int(file)
            
        if file.startswith('/dev/'):
            return file
            
        if not ('/' in file or '\\' in file):
            filename = secure_filename(file)
            file_path = os.path.join(Config.VIDEO_FOLDER, filename)
            return file_path
            
        return file
