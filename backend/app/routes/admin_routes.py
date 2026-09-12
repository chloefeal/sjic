"""
运维管理：远程重启后端 / 边缘 Agent（客户管理员 admin）
"""
import os
import threading
import time
from flask import Blueprint, jsonify, current_app, request
from app.middleware.auth import token_required, role_required, get_token_payload
from app.models.edge_node import EdgeNode
from app.services.mqtt_service import mqtt_service

admin_bp = Blueprint('admin', __name__)

_backend_restart_lock = threading.Lock()
_backend_restart_pending = False


@admin_bp.route('/api/admin/restart-backend', methods=['POST'])
@token_required
@role_required('customer')
def restart_backend():
    """
    重启管理平台后端进程。
    Docker Compose 下 PID1 退出后会由 restart: unless-stopped 自动拉起。
    本地 python run.py 开发模式不会自动拉起，需手动再启。
    """
    global _backend_restart_pending

    if not _backend_restart_lock.acquire(blocking=False):
        return jsonify({'error': 'Restart already in progress'}), 409

    try:
        if _backend_restart_pending:
            return jsonify({'error': 'Restart already in progress'}), 409
        _backend_restart_pending = True

        payload = get_token_payload() or {}
        current_app.logger.warning(
            "Backend restart requested by user=%s role=%s",
            payload.get('user'),
            payload.get('role'),
        )

        app = current_app._get_current_object()

        def _exit_soon():
            time.sleep(1.2)
            try:
                app.logger.warning("Exiting process for backend restart")
            except Exception:
                pass
            os._exit(0)

        threading.Thread(target=_exit_soon, daemon=True).start()
        return jsonify({
            'message': 'Backend is restarting',
            'hint': 'Docker 部署约数秒后自动恢复；开发环境请手动重启后端进程。',
        })
    finally:
        _backend_restart_lock.release()


@admin_bp.route('/api/nodes/<int:node_id>/restart', methods=['POST'])
@token_required
@role_required('customer')
def restart_edge_node(node_id):
    """通过 MQTT 通知边缘 Agent 优雅退出；Docker Compose restart: unless-stopped 自动拉起。"""
    node = EdgeNode.query.get(node_id)
    if not node:
        return jsonify({'error': 'Node not found'}), 404

    reason = (request.get_json(silent=True) or {}).get('reason') or 'admin'
    payload = get_token_payload() or {}
    current_app.logger.warning(
        "Edge agent restart requested node_id=%s mac=%s by user=%s reason=%s",
        node.id,
        node.mac_address,
        payload.get('user'),
        reason,
    )

    try:
        mqtt_service.publish_agent_restart(node.mac_address, reason=reason)
    except Exception as e:
        current_app.logger.error(f"Failed to publish agent restart: {e}")
        return jsonify({'error': f'Failed to publish restart command: {e}'}), 500

    return jsonify({
        'message': f'Restart command sent to {node.name}',
        'node_id': node.id,
        'mac_address': node.mac_address,
        'hint': '边缘端需用 Docker Compose（restart: unless-stopped）运行，进程退出后会自动拉起。',
    })
