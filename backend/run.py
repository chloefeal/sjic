from app import app, socketio, db
from config import Config

def init_db():
    app.app_context().push()
    db.create_all()

if __name__ == '__main__':
    init_db()  # 初始化数据库
    socketio.run(app, 
        host='0.0.0.0',  # 确保监听所有网络接口
        port=Config.BACKEND_PORT, 
        debug=True,
        allow_unsafe_werkzeug=True  # 允许在生产环境使用werkzeug
    ) 