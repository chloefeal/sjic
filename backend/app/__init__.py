from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
socketio = SocketIO(app, cors_allowed_origins="*")

def init_app():
    with app.app_context():
        # 导入路由和模型
        from app.routes import api_routes
        from app.models import camera, detection_model, alert, settings, log
        
        # 创建数据库表
        db.create_all()
        
        return app

# 初始化应用
init_app() 