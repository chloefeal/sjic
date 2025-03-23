from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from config import Config
import logging
from logging.handlers import RotatingFileHandler
import os
from flask_sock import Sock
import tempfile
import shutil

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

# 配置 CORS
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "expose_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "max_age": 86400
    },
    r"/ws/*": {"origins": "*"}
})

db = SQLAlchemy(app)
migrate = Migrate(app, db)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    logger=True,
    engineio_logger=True
)

# 初始化 Flask-Sock
sock = Sock(app)

# 检查临时目录
temp_dir = tempfile.gettempdir()
app.logger.info(f"Using temporary directory: {temp_dir}")

# 确保临时目录可写
if not os.access(temp_dir, os.W_OK):
    app.logger.warning(f"Temporary directory {temp_dir} is not writable!")
    
# 检查磁盘空间
total, used, free = shutil.disk_usage(temp_dir)
app.logger.info(f"Disk space: total={total//(1024**3)}GB, used={used//(1024**3)}GB, free={free//(1024**3)}GB")

def init_app():
    with app.app_context():
        app.logger.info('Initializing application...')
        
        # 导入路由和模型
        from app.routes import api_routes, auth_routes
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