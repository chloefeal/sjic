from app import app, db
from config import Config

def init_db():
    app.app_context().push()
    db.create_all()

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=Config.BACKEND_PORT, debug=True) 