from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from config import Config
import logging
from logging.handlers import RotatingFileHandler
import os

# 创建日志目录
os.makedirs(Config.LOG_FOLDER, exist_ok=True)

# 配置日志处理器
formatter = logging.Formatter(Config.LOG_FORMAT)

# 文件处理器
file_handler = RotatingFileHandler(
    Config.LOG_PATH,
    maxBytes=Config.LOG_MAX_BYTES,
    backupCount=Config.LOG_BACKUP_COUNT
)
file_handler.setFormatter(formatter)
file_handler.setLevel(Config.LOG_LEVEL)

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(Config.LOG_LEVEL)

# 配置Flask应用的日志
app = Flask(__name__)
app.logger.addHandler(file_handler)
app.logger.addHandler(console_handler)
app.logger.setLevel(Config.LOG_LEVEL)

app.config.from_object(Config)

# 配置 CORS - 允许所有路由的 OPTIONS 请求
CORS(app, 
    resources={r"/api/*": {
        "origins": "*",  # 允许所有域名访问
        "methods": Config.CORS_METHODS,
        "allow_headers": Config.CORS_HEADERS,
        "supports_credentials": True,
        "max_age": 1728000  # 预检请求缓存20天
    }},
    expose_headers=["Content-Type", "Authorization"]
)

db = SQLAlchemy(app)
migrate = Migrate(app, db)
socketio = SocketIO(app, cors_allowed_origins="*")

def init_app():
    with app.app_context():
        app.logger.info('Initializing application...')
        
        # 导入路由和模型
        from app.routes import api_routes
        from app.models import camera, detection_model, alert, task, log
        from app.algorithms.base import BaseAlgorithm
        
        # 创建数据库表
        db.create_all()
        app.logger.info('Database tables created')
        
        # 注册算法
        BaseAlgorithm.register_algorithms()
        app.logger.info('Algorithms registered')
        
        return app

# 初始化应用
init_app() 