from flask import jsonify, request
from app import app
import jwt
from datetime import datetime, timedelta
from config import Config

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    # 验证默认账号密码
    if username == 'admin' and password == '123123':
        token = jwt.encode({
            'user': username,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, Config.SECRET_KEY, algorithm='HS256')
        
        return jsonify({
            'token': token,
            'username': username
        })
    
    return jsonify({'error': 'Invalid credentials'}), 401 