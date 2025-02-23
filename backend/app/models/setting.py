from app import db, app
from datetime import datetime

class Setting(db.Model):
    __tablename__ = 'settings'
    
    id = db.Column(db.Integer, primary_key=True)
    config = db.Column(db.JSON, nullable=False, default=dict)  # 存储所有配置项
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 默认配置
    DEFAULT_CONFIG = {
        'external_alert_api': {
            'url': '',
            'token': '',
            'secret': ''
        },
        'alert': {
            'retention_days': 30,
            'image_quality': 95
        },
        'system': {
            'log_level': 'INFO'
        }
    }

    def __init__(self):
        super().__init__()
        self.config = self.DEFAULT_CONFIG.copy()

    def to_dict(self):
        return self.config

    def update(self, data):
        """更新设置"""
        # 递归更新配置
        def update_dict(current, new):
            for key, value in new.items():
                if key in current:
                    if isinstance(value, dict) and isinstance(current[key], dict):
                        update_dict(current[key], value)
                    else:
                        app.logger.warning(f"invalid key: {key} with value: {value}")
                        continue

        update_dict(self.config, data)
