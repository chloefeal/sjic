"""任务检测预览：平台记住最新一帧，并通知边缘开始或停止回传。"""
import base64
import threading
import time

from app.models.algorithm import Algorithm
from app.models.detection_model import DetectionModel

_PREVIEW_TTL = 12
_MAX_IMAGE_BYTES = 1_500_000


def normalize_labelmap(raw):
    out = {}
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = []
        for key, value in raw.items():
            if isinstance(value, dict):
                items.append(value)
            else:
                items.append({'id': key, 'name': value})
    else:
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            class_id = int(item.get('id'))
        except (TypeError, ValueError):
            continue
        name = str(item.get('name') or '').strip()
        if name:
            out[class_id] = name
    return out


def preview_engines(task, engine=None):
    if task.is_exam_pipeline():
        engines = []
        if task.od_algorithm_id:
            engines.append('object_detection')
        if task.pose_algorithm_id:
            engines.append('pose_behavior')
        return engines
    resolved = engine
    if resolved is None and task.algorithm_id:
        algo = Algorithm.query.get(task.algorithm_id)
        resolved = algo.resolved_engine() if algo else ''
    if resolved in ('object_detection', 'pose_behavior'):
        return [resolved]
    return []


def object_labelmap(task, engine=None):
    """只取目标检测模型的类别名。姿态预览固定标成「人」。"""
    if 'object_detection' not in preview_engines(task, engine):
        return {}
    algo_id = task.od_algorithm_id if task.is_exam_pipeline() else task.algorithm_id
    if not algo_id:
        return {}
    algo = Algorithm.query.get(algo_id)
    if not algo or not algo.model_id:
        return {}
    model = DetectionModel.query.get(algo.model_id)
    if not model:
        return {}
    named = normalize_labelmap(model.labelmap)
    return {str(class_id): name for class_id, name in named.items()}


class PreviewHub:
    """每个打开的预览窗口一个 view。最后一个窗口关掉后不再接收画面。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._views = {}
        self._frames = {}

    def set_view(self, task_id, view, enabled, issued_at, ttl=_PREVIEW_TTL):
        """返回 on / off / extend / ignored。issued_at 用来丢掉先发后到的旧请求。"""
        task_key = int(task_id)
        view_key = str(view or '').strip()[:64]
        if not view_key:
            return 'ignored'
        try:
            issued = int(issued_at)
        except (TypeError, ValueError):
            issued = 0
        now = time.time()
        with self._lock:
            bucket = self._views.setdefault(task_key, {})
            prev = bucket.get(view_key)
            if prev and issued < prev[0]:
                return 'ignored'
            before = self._is_live_locked(task_key, now)
            if enabled:
                bucket[view_key] = (issued, now + float(ttl))
            else:
                bucket.pop(view_key, None)
            self._prune_locked(task_key, now)
            after = self._is_live_locked(task_key, now)
            if not after:
                self._frames.pop(task_key, None)
            if not before and after:
                return 'on'
            if before and not after:
                return 'off'
            return 'extend'

    def is_live(self, task_id):
        now = time.time()
        with self._lock:
            self._prune_locked(int(task_id), now)
            return self._is_live_locked(int(task_id), now)

    def ingest(self, task_id, image, stats):
        if not image or len(image) > _MAX_IMAGE_BYTES:
            return False
        if not isinstance(stats, dict):
            stats = {}
        task_key = int(task_id)
        now = time.time()
        with self._lock:
            self._prune_locked(task_key, now)
            if not self._is_live_locked(task_key, now):
                return False
            self._frames[task_key] = {
                'image': bytes(image),
                'stats': stats,
                'updated_at': now,
            }
            return True

    def snapshot(self, task_id):
        task_key = int(task_id)
        now = time.time()
        with self._lock:
            self._prune_locked(task_key, now)
            if not self._is_live_locked(task_key, now):
                self._frames.pop(task_key, None)
                return {'waiting': True, 'age_ms': None, 'stats': {}, 'image': None}
            frame = self._frames.get(task_key)
            if not frame:
                return {'waiting': True, 'age_ms': None, 'stats': {}, 'image': None}
            age_ms = max(0, int((now - frame['updated_at']) * 1000))
            return {
                'waiting': False,
                'age_ms': age_ms,
                'stats': frame['stats'] or {},
                'image': base64.b64encode(frame['image']).decode('ascii'),
            }

    def _prune_locked(self, task_id, now):
        bucket = self._views.get(task_id)
        if not bucket:
            return
        expired = [key for key, (_, until) in bucket.items() if until <= now]
        for key in expired:
            bucket.pop(key, None)
        if not bucket:
            self._views.pop(task_id, None)

    def _is_live_locked(self, task_id, now):
        bucket = self._views.get(task_id) or {}
        return any(until > now for _, until in bucket.values())


preview_hub = PreviewHub()
