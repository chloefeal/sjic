"""
首页概览统计路由
"""
from flask import Blueprint, jsonify, current_app
from app.extensions import db
from app.models.camera import Camera
from app.models.detection_model import DetectionModel
from app.models.algorithm import Algorithm
from app.models.edge_node import EdgeNode
from app.models.task import Task
from app.models.alert import Alert
from app.middleware.auth import token_required

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/api/dashboard/summary', methods=['GET'])
@token_required
def dashboard_summary():
    """
    首页概述统计：资源数量 + 近期告警滚动列表 + 最新告警（含截图）
    """
    try:
        camera_count = Camera.query.count()
        model_count = DetectionModel.query.count()
        algorithm_count = Algorithm.query.count()
        node_count = EdgeNode.query.count()
        online_node_count = EdgeNode.query.filter_by(status='online').count()
        task_count = Task.query.count()
        running_task_count = Task.query.filter(
            db.or_(
                Task.run_status == 'running',
                Task.status == 'running',
            )
        ).count()
        alert_total = Alert.query.count()

        recent_alerts = (
            Alert.query.order_by(Alert.timestamp.desc())
            .limit(20)
            .all()
        )
        latest_alert = recent_alerts[0].to_dict() if recent_alerts else None

        return jsonify({
            'counts': {
                'cameras': camera_count,
                'models': model_count,
                'algorithms': algorithm_count,
                'nodes': node_count,
                'nodes_online': online_node_count,
                'tasks': task_count,
                'tasks_running': running_task_count,
                'alerts': alert_total,
            },
            'recent_alerts': [a.to_dict() for a in recent_alerts],
            'latest_alert': latest_alert,
        })
    except Exception as e:
        current_app.logger.error(f"Error building dashboard summary: {str(e)}")
        return jsonify({'error': str(e)}), 500
