from app import app, db, socketio, sock
from config import Config

def init_db():
    app.app_context().push()
    db.create_all()

if __name__ == '__main__':
    init_db()
    socketio.run(app, host='0.0.0.0', port=38881, debug=True, allow_unsafe_werkzeug=True) 