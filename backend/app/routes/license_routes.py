"""
License management routes.
"""
from flask import Blueprint, jsonify, request
from app.middleware.auth import token_required
from app.middleware.license_guard import REASON_MESSAGES
from app.services.license_service import license_service


license_bp = Blueprint('license', __name__)


def _enrich(status):
    reason = status.get('reason') or ''
    status = dict(status)
    status['message'] = REASON_MESSAGES.get(reason, reason)
    return status


@license_bp.route('/api/license/status', methods=['GET'])
@token_required
def get_license_status():
    return jsonify(_enrich(license_service.get_status())), 200


@license_bp.route('/api/license/import', methods=['POST'])
@token_required
def import_license():
    """
    导入授权文件（JSON）。
    支持 multipart 文件字段 license / file，或 JSON body。
    """
    raw = None

    if request.files:
        uploaded = request.files.get('license') or request.files.get('file')
        if uploaded:
            raw = uploaded.read().decode('utf-8', errors='replace')

    if raw is None:
        if request.is_json:
            body = request.get_json(silent=True)
            if isinstance(body, dict) and 'license' in body and isinstance(body['license'], (dict, str)):
                raw = body['license']
            else:
                raw = body
        else:
            raw = request.get_data(as_text=True)

    if not raw:
        return jsonify({'error': '缺少授权内容', 'reason': 'invalid_json'}), 400

    ok, reason, status = license_service.import_license(raw)
    if not ok:
        return jsonify({
            'error': REASON_MESSAGES.get(reason, reason or '导入失败'),
            'reason': reason,
            'status': _enrich(status),
        }), 400

    return jsonify({
        'message': '授权导入成功',
        'status': _enrich(status),
    }), 200
