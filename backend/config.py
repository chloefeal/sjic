import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # 文件存储配置
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    
    # 模型配置
    MODEL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    
    # 日志配置
    LOG_FOLDER = 'logs'
    
    # 区域配置存储
    REGION_CONFIG_PATH = 'data/regions.json'
    
    # 端口配置
    #FRONTEND_PORT = 38880
    BACKEND_PORT = 38881 