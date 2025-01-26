from app import app, socketio, db

def init_db():
    app.app_context().push()
    db.create_all()

if __name__ == '__main__':
    init_db()  # 初始化数据库
    socketio.run(app, host='0.0.0.0', port=5000, debug=True) 