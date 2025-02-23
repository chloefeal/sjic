from flask import jsonify, request, Response
from app import app, db, socketio
from app.models import Alert, Camera, DetectionModel, Algorithm, Setting
from app.services.detector import DetectorService
from app.services.model_trainer import ModelTrainer
from app.models import Task, Log
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from config import Config
from app.middleware.auth import token_required
import cv2
import time
from app.utils.calibration import get_calibration_image
from sqlalchemy.orm.attributes import flag_modified


# 初始化服务
detector_service = DetectorService()
model_trainer = ModelTrainer()

# 摄像头相关路由
@app.route('/api/cameras', methods=['GET'])
@token_required
def get_cameras():
    """获取所有摄像头"""
    try:
        cameras = Camera.query.all()
        return jsonify([camera.to_dict() for camera in cameras])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cameras', methods=['POST'])
@token_required
def create_camera():
    """创建新摄像头"""
    try:
        data = request.json
        app.logger.info(f"Creating new camera: {data}")
        
        camera = Camera(
            name=data['name'],
            url=data['url']
        )
        db.session.add(camera)
        db.session.commit()
        
        app.logger.info(f"Camera created successfully: id={camera.id}")
        return jsonify(camera.to_dict()), 201
    except Exception as e:
        app.logger.error(f"Failed to create camera: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/cameras/<int:camera_id>', methods=['DELETE'])
@token_required
def delete_camera(camera_id):
    """删除摄像头"""
    camera = Camera.query.get_or_404(camera_id)
    db.session.delete(camera)
    db.session.commit()
    return '', 204

# 模型相关路由
@app.route('/api/models', methods=['GET'])
@token_required
def get_models():
    """获取所有模型"""
    models = DetectionModel.query.all()
    return jsonify([model.to_dict() for model in models])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@app.route('/api/models/upload', methods=['POST'])
@token_required
def upload_model():
    """上传新模型"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # 确保模型目录存在
        Path(Config.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
        
        # 安全的文件名
        filename = secure_filename(file.filename)
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        
        # 保存文件
        file.save(file_path)
        
        # 创建模型记录
        model = DetectionModel(
            name=request.form.get('name', filename),
            path=filename,
            description=request.form.get('description', '')
        )
        db.session.add(model)
        db.session.commit()
        
        return jsonify(model.to_dict()), 201
        
    except Exception as e:
        # 如果出错，清理已上传的文件
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<int:model_id>', methods=['DELETE'])
@token_required 
def delete_model(model_id):
    """删除模型"""
    model = DetectionModel.query.get_or_404(model_id)
    db.session.delete(model)
    db.session.commit()
    return '', 204

# 检测相关路由
@app.route('/api/detection/start', methods=['POST'])
@token_required 
def start_detection():
    """启动检测"""
    data = request.json
    try:
        detector_service.start(
            camera_id=data['camera_id'],
            model_id=data['model_id'],
            task_id=data['task_id']
        )
        return jsonify({'message': 'Detection started'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/detection/stop', methods=['POST'])
@token_required 
def stop_detection():
    """停止检测"""
    data = request.json
    try:
        detector_service.stop(data['camera_id'])
        return jsonify({'message': 'Detection stopped'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# 告警相关路由
@app.route('/api/alerts', methods=['GET'])
@token_required 
def get_alerts():
    """获取告警记录"""
    alerts = Alert.query.order_by(Alert.timestamp.desc()).all()
    return jsonify([alert.to_dict() for alert in alerts])

@app.route('/api/alerts', methods=['POST'])
@token_required 
def create_alert():
    """创建新告警"""
    data = request.json
    
    camera = Camera.query.get(data['camera_id'])
    if not camera:
        return jsonify({'error': 'Camera not found'}), 404
        
    alert = Alert(
        camera_id=data['camera_id'],
        alert_type=data['alert_type'],
        confidence=data.get('confidence'),
        image_url=data.get('image_url')
    )
    
    db.session.add(alert)
    db.session.commit()
    
    # 通过WebSocket发送实时告警
    socketio.emit('new_alert', alert.to_dict())
    
    return jsonify(alert.to_dict()), 201

@app.route('/api/alerts/<int:alert_id>', methods=['DELETE'])
@token_required 
def delete_alert(alert_id):
    """删除告警记录"""
    alert = Alert.query.get_or_404(alert_id)
    db.session.delete(alert)
    db.session.commit()
    return '', 204

# 训练相关路由
@app.route('/api/training/start', methods=['POST'])
@token_required 
def start_training():
    """开始模型训练"""
    if 'dataset' not in request.files:
        return jsonify({'error': 'No dataset provided'}), 400
        
    dataset = request.files['dataset']
    config = request.form.get('config', '{}')
    
    try:
        result = model_trainer.train(dataset, config)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# 错误处理
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/algorithms', methods=['GET'])
@token_required 
def get_algorithms():
    """获取算法"""
    algorithms = Algorithm.query.all()
    return jsonify([algorithm.to_dict() for algorithm in algorithms])

@app.route('/api/algorithms', methods=['POST'])
@token_required 
def create_algorithm():
    """创建新算法"""
    data = request.json
    algorithm = Algorithm(**data)
    db.session.add(algorithm)
    db.session.commit()
    return jsonify(algorithm.to_dict()), 201

@app.route('/api/tasks', methods=['GET'])
@token_required 
def get_tasks():
    """获取算法设置"""
    tasks = Task.query.all()
    return jsonify([task.to_dict() for task in tasks])

@app.route('/api/tasks', methods=['POST'])
@token_required 
def create_tasks():
    """创建算法设置"""
    data = request.json
    app.logger.info(f"Creating new task: {data}")
    task = Task(**data)
    task.save_calibration_image()
    db.session.add(task)
    db.session.commit()
    return jsonify(task.to_dict()), 201

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
@token_required
def update_tasks(task_id):
    """更新算法设置"""
    try:
        data = request.json
        #app.logger.info(f"Updating task: {data}")
        task = Task.query.get_or_404(task_id)
        for key, value in data.items():
            if hasattr(task, key):
                setattr(task, key, value)
        task.save_calibration_image()

        # JSON格式字段需要明确地标记字段已修改
        flag_modified(task, 'algorithm_parameters')
        
        db.session.commit()
        return jsonify(task.to_dict())
        
    except Exception as e:
        app.logger.error(f"Error updating task: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
@token_required 
def delete_tasks(task_id):
    """删除算法设置"""
    task = Task.query.get_or_404(task_id)
    db.session.delete(task)
    db.session.commit()
    return '', 204

@app.route('/api/tasks/<int:task_id>/detail', methods=['GET'])
@token_required 
def get_task_detail(task_id):
    task = Task.query.get_or_404(task_id)
    
    # 获取任务详情，包括标定图像
    task_data = task.to_dict()
    
    # 如果有标定图像，添加图像数据
    if task.algorithm_parameters and 'calibration' in task.algorithm_parameters:
        calibration = task.algorithm_parameters['calibration']
        if 'image_path' in calibration:
            # 获取图像数据
            image_data = get_calibration_image(task_id)
            if image_data:
                calibration['image_data'] = image_data
    
    return jsonify(task_data)

@app.route('/api/logs', methods=['GET'])
@token_required
def get_logs():
    """获取系统日志"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    logs = Log.query.order_by(Log.timestamp.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    return jsonify({
        'items': [log.to_dict() for log in logs.items],
        'total': logs.total,
        'pages': logs.pages,
        'current_page': logs.page
    })

@app.route('/api/logs', methods=['DELETE'])
@token_required
def clear_logs():
    """清除所有日志"""
    Log.query.delete()
    db.session.commit()
    return '', 204

@app.route('/api/logs/delete/<int:id>', methods=['DELETE'])
@token_required
def delete_log(id):
    log = Log.query.get(id)
    db.session.delete(log)
    db.session.commit()
    return jsonify({"status": "success"})

@app.route('/api/cameras/capture', methods=['POST'])
def capture_frame():
    """获取摄像头当前帧"""
    try:
        data = request.json
        camera_id = data.get('camera_id')
        
        # 获取摄像头
        camera = Camera.query.get_or_404(camera_id)
        cap = cv2.VideoCapture(camera.get_rtsp_url())
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {camera_id}")
            
        # 读取一帧
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError("Failed to capture frame")
            
        # 将图片编码为JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        
        # 确保返回正确的 MIME 类型和二进制数据
        return Response(
            buffer.tobytes(),
            mimetype='image/jpeg',
            headers={
                'Content-Type': 'image/jpeg',
                'Content-Disposition': 'inline'
            }
        )
    except Exception as e:
        app.logger.error(f"Capture frame error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """处理连接"""
    app.logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """处理断开连接"""
    app.logger.info('Client disconnected')

@socketio.on('start_stream')
def handle_start_stream(data):
    """处理开始流请求"""
    cap = None
    try:
        camera_id = data.get('camera_id')
        camera = Camera.query.get_or_404(camera_id)
        cap = cv2.VideoCapture(camera.get_rtsp_url())
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 将帧编码为JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            
            # 发送帧到客户端
            socketio.emit('frame', frame_data, room=request.sid)
            
            # 控制帧率
            socketio.sleep(1/30)  # 30fps
            
    except Exception as e:
        app.logger.error(f"Stream error: {str(e)}")
    finally:
        if cap:
            cap.release()

@app.route('/api/settings', methods=['GET'])
@token_required
def get_settings():
    """获取系统设置"""
    try:
        settings = Setting.query.first()
        if not settings:
            settings = Setting()  # 使用默认值
            db.session.add(settings)
            db.session.commit()
        return jsonify(settings.to_dict())
    except Exception as e:
        app.logger.error(f"Error getting settings: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['POST'])
@token_required
def update_settings():
    """更新系统设置"""
    try:
        data = request.get_json()
        app.logger.info(f"Received settings update: {data}")
        
        settings = Setting.query.first()
        if not settings:
            settings = Setting()
            db.session.add(settings)
        
        # 更新设置
        settings.update(data)
        
        # 提交更改
        try:
            db.session.commit()
            app.logger.info("Settings committed to database")
            
            # 验证更新
            updated_settings = Setting.query.get(settings.id)
            app.logger.info(f"Verified settings after commit: {updated_settings.config}")
            
            return jsonify({'message': 'Settings updated successfully'})
        except Exception as e:
            app.logger.error(f"Error committing settings: {str(e)}")
            db.session.rollback()
            raise
            
    except Exception as e:
        app.logger.error(f"Error updating settings: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500



