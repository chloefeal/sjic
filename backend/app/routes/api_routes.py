from flask import jsonify, request
from app import app, db, socketio
from app.models import Alert, Camera, DetectionModel, Algorithm
from app.services.detector import DetectorService
from app.services.model_trainer import ModelTrainer
from app.models import Task, Log
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from config import Config

# 初始化服务
detector_service = DetectorService()
model_trainer = ModelTrainer()

# 摄像头相关路由
@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """获取所有摄像头"""
    try:
        cameras = Camera.query.all()
        return jsonify([camera.to_dict() for camera in cameras])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cameras', methods=['POST'])
def create_camera():
    """创建新摄像头"""
    data = request.json
    camera = Camera(
        name=data['name'],
        url=data['url']
    )
    db.session.add(camera)
    db.session.commit()
    return jsonify(camera.to_dict()), 201

@app.route('/api/cameras/<int:camera_id>', methods=['DELETE'])
def delete_camera(camera_id):
    """删除摄像头"""
    camera = Camera.query.get_or_404(camera_id)
    db.session.delete(camera)
    db.session.commit()
    return '', 204

# 模型相关路由
@app.route('/api/models', methods=['GET'])
def get_models():
    """获取所有模型"""
    models = DetectionModel.query.all()
    return jsonify([model.to_dict() for model in models])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@app.route('/api/models/upload', methods=['POST'])
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
def delete_model(model_id):
    """删除模型"""
    model = DetectionModel.query.get_or_404(model_id)
    db.session.delete(model)
    db.session.commit()
    return '', 204

# 检测相关路由
@app.route('/api/detection/start', methods=['POST'])
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
def get_alerts():
    """获取告警记录"""
    alerts = Alert.query.order_by(Alert.timestamp.desc()).all()
    return jsonify([alert.to_dict() for alert in alerts])

@app.route('/api/alerts', methods=['POST'])
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
def delete_alert(alert_id):
    """删除告警记录"""
    alert = Alert.query.get_or_404(alert_id)
    db.session.delete(alert)
    db.session.commit()
    return '', 204

# 训练相关路由
@app.route('/api/training/start', methods=['POST'])
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
def get_algorithms():
    """获取算法"""
    algorithms = Algorithm.query.all()
    return jsonify([algorithm.to_dict() for algorithm in algorithms])

@app.route('/api/algorithms', methods=['POST'])
def create_algorithm():
    """创建新算法"""
    data = request.json
    algorithm = Algorithm(**data)
    db.session.add(algorithm)
    db.session.commit()
    return jsonify(algorithm.to_dict()), 201

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """获取算法设置"""
    tasks = Task.query.all()
    return jsonify([task.to_dict() for task in tasks])

@app.route('/api/tasks', methods=['POST'])
def create_tasks():
    """创建算法设置"""
    data = request.json
    task = Task(**data)
    db.session.add(task)
    db.session.commit()
    return jsonify(task.to_dict()), 201

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_tasks(task_id):
    """更新算法设置"""
    data = request.json
    task = Task.query.get_or_404(task_id)
    for key, value in data.items():
        if hasattr(task, key):
            setattr(task, key, value)
    db.session.commit()
    return jsonify(task.to_dict())

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_tasks(task_id):
    """删除算法设置"""
    task = Task.query.get_or_404(task_id)
    db.session.delete(task)
    db.session.commit()
    return '', 204

@app.route('/api/logs', methods=['GET'])
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
def clear_logs():
    """清除所有日志"""
    Log.query.delete()
    db.session.commit()
    return '', 204

@app.route('/api/logs/delete/<int:id>', methods=['DELETE'])
def delete_log(id):
    log = Log.query.get(id)
    db.session.delete(log)
    db.session.commit()
    return jsonify({"status": "success"})

