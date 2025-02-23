from app import db
from datetime import datetime

class Setting(db.Model):
    __tablename__ = 'settings'
    
    id = db.Column(db.Integer, primary_key=True)
    external_api_url = db.Column(db.String(200))
    external_api_token = db.Column(db.String(200))
    external_api_secret = db.Column(db.String(200))
    alert_retention_days = db.Column(db.Integer, default=30)
    alert_image_quality = db.Column(db.Integer, default=95)
    log_level = db.Column(db.String(20), default='INFO')
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'external_api': {
                'url': self.external_api_url,
                'token': self.external_api_token,
                'secret': self.external_api_secret
            },
            'alert': {
                'retention_days': self.alert_retention_days,
                'image_quality': self.alert_image_quality
            },
            'system': {
                'log_level': self.log_level,
            }
        }

    def update(self, data):
        """更新设置"""
        external_api = data.get('external_api', {})
        self.external_api_url = external_api.get('url', self.external_api_url)
        self.external_api_token = external_api.get('token', self.external_api_token)
        self.external_api_secret = external_api.get('secret', self.external_api_secret)
        
        alert = data.get('alert', {})
        self.alert_retention_days = alert.get('retention_days', self.alert_retention_days)
        self.alert_image_quality = alert.get('image_quality', self.alert_image_quality)
        
        system = data.get('system', {})
        self.log_level = system.get('log_level', self.log_level)
