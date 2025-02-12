from flask import jsonify, request, make_response
from app import app
from flask_cors import cross_origin
import jwt
from datetime import datetime, timedelta
from config import Config

@app.route('/api/login', methods=['POST', 'OPTIONS'])
@cross_origin(origins="*", methods=['POST', 'OPTIONS'], supports_credentials=True)
def login():
    app.logger.info(f"Login request received: method={request.method}")
    
    # 处理预检请求
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
        return response

    # 处理实际的登录请求
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    app.logger.info(f"Processing login for username: {username}")
    
    # 验证账号密码
    if username == 'admin' and password == '123123':
        token = jwt.encode({
            'user': username,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, Config.SECRET_KEY, algorithm='HS256')
        
        response = jsonify({
            'token': token,
            'username': username
        })
        return response
    
    return jsonify({'error': 'Invalid credentials'}), 401 