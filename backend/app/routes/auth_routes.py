from flask import jsonify, request
from app import app
from flask_cors import cross_origin
import jwt
from datetime import datetime, timedelta
from config import Config

@app.route('/api/login', methods=['POST', 'OPTIONS'])
@cross_origin(origins="*", methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return '', 200
        
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    app.logger.info(f"Login request received: username={username}")
    
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