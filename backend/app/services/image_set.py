"""图片测试集：磁盘清单，不进数据库。

边缘把 `image-set:<id>` 当成一路视频，反复读取「当前图片」。
"""
import json
import os
import re
import threading
import uuid

from flask import current_app

IMAGE_SET_PREFIX = 'image-set:'
MAX_IMAGES = 500
MAX_BYTES = 25 * 1024 * 1024
MIN_SIDE = 16
MAX_SIDE = 8000
_SET_ID_RE = re.compile(r'^[A-Za-z0-9_-]{1,32}$')
_STORED_RE = re.compile(r'^[a-f0-9]{16}\.(jpg|png|bmp|webp)$')
_EXT_MIME = {
    '.jpg': 'image/jpeg',
    '.png': 'image/png',
    '.bmp': 'image/bmp',
    '.webp': 'image/webp',
}
_LOCK = threading.Lock()
_JPEG_CACHE = {}


class ImageSetError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def camera_url(set_id):
    validate_set_id(set_id)
    return f'{IMAGE_SET_PREFIX}{set_id}'


def validate_set_id(set_id):
    if not _SET_ID_RE.fullmatch(set_id or ''):
        raise ImageSetError('无效的图片集')


def parse_set_id(url):
    raw = str(url or '').strip()
    if not raw.startswith(IMAGE_SET_PREFIX):
        raise ImageSetError('不是图片测试流')
    set_id = raw[len(IMAGE_SET_PREFIX):].strip()
    validate_set_id(set_id)
    return set_id


def snapshot(set_id):
    validate_set_id(set_id)
    with _LOCK:
        return _public(set_id, _load(set_id))


def edge_meta(set_id):
    data = snapshot(set_id)
    return {
        'id': data['id'],
        'current': data['current'],
        'revision': data['revision'],
    }


def add_images(set_id, items):
    """items: [{original, raw}]。能解码的写入，其余放进 errors。"""
    validate_set_id(set_id)
    if not items:
        raise ImageSetError('没有选择图片')

    prepared = []
    errors = []
    for item in items:
        original = _display_name(item.get('original'))
        raw = item.get('raw') or b''
        try:
            prepared.append(_prepare(original, raw))
        except ImageSetError as exc:
            errors.append({'filename': original, 'error': str(exc)})

    if not prepared:
        message = errors[0]['error'] if errors else '没有可用图片'
        raise ImageSetError(message)

    accepted = []

    def fn(data):
        room = MAX_IMAGES - len(data['images'])
        if room <= 0:
            raise ImageSetError(f'图片数量不能超过 {MAX_IMAGES}')
        take = prepared[:room]
        for item in prepared[room:]:
            errors.append({'filename': item['original'], 'error': f'图片数量不能超过 {MAX_IMAGES}'})
        for item in take:
            _write_bytes(set_id, item['name'], item['raw'])
            row = {
                'name': item['name'],
                'label': '',
                'original': item['original'],
            }
            data['images'].append(row)
            accepted.append(row)

    public = _mutate(set_id, fn)
    public['errors'] = errors
    public['added'] = accepted
    return public


def set_label(set_id, name, label):
    if label not in ('', 'normal', 'abnormal'):
        raise ImageSetError('标记只能是正常、异常或空')
    _require_stored_name(name)

    def fn(data):
        for item in data['images']:
            if item.get('name') == name:
                item['label'] = label
                return
        raise ImageSetError('图片不存在', 404)

    return _mutate(set_id, fn)


def delete_image(set_id, name):
    _require_stored_name(name)

    def fn(data):
        kept = [item for item in data['images'] if item.get('name') != name]
        if len(kept) == len(data['images']):
            raise ImageSetError('图片不存在', 404)
        data['images'] = kept
        if data.get('current') == name:
            data['current'] = None
        path = _stored_path(set_id, name)
        if os.path.isfile(path):
            os.remove(path)

    return _mutate(set_id, fn)


def set_current(set_id, name):
    _require_stored_name(name)

    def fn(data):
        if not any(item.get('name') == name for item in data['images']):
            raise ImageSetError('图片不存在', 404)
        data['current'] = name

    return _mutate(set_id, fn)


def stored_file(set_id, name):
    validate_set_id(set_id)
    _require_stored_name(name)
    path = _stored_path(set_id, name)
    if not os.path.isfile(path):
        raise ImageSetError('图片不存在', 404)
    return path, _mime(name)


def current_file(set_id):
    data = snapshot(set_id)
    name = data.get('current')
    if not name:
        raise ImageSetError('尚未选择当前图片', 404)
    return stored_file(set_id, name)


def current_jpeg(set_id, max_width=0):
    """预览用 JPEG。没有当前图时返回 None。"""
    validate_set_id(set_id)
    with _LOCK:
        data = _load(set_id)
        name = data.get('current') or None
        revision = int(data.get('revision') or 0)
    if not name:
        return None
    try:
        path, _mime_type = stored_file(set_id, name)
    except ImageSetError:
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    key = (set_id, name, revision, int(max_width or 0), mtime)
    cached = _JPEG_CACHE.get(key)
    if cached is not None:
        return cached
    with open(path, 'rb') as handle:
        raw = handle.read()
    frame = decode_bgr(raw)
    if frame is None:
        return None
    if max_width and frame.shape[1] > max_width:
        import cv2
        scale = max_width / float(frame.shape[1])
        size = (int(max_width), max(1, int(frame.shape[0] * scale)))
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    jpeg = _encode_jpeg(frame)
    if jpeg is None:
        return None
    _JPEG_CACHE.clear()
    _JPEG_CACHE[key] = jpeg
    return jpeg


def decode_bgr(raw):
    if not raw or len(raw) > MAX_BYTES:
        return None
    import cv2
    import numpy as np
    arr = np.frombuffer(bytearray(raw), dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None or frame.ndim != 3:
        return None
    height, width = frame.shape[:2]
    if height < MIN_SIDE or width < MIN_SIDE or height > MAX_SIDE or width > MAX_SIDE:
        return None
    return frame


def _encode_jpeg(frame):
    import cv2
    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    return buf.tobytes()


def _prepare(original, raw):
    if not raw:
        raise ImageSetError('文件是空的')
    if len(raw) > MAX_BYTES:
        raise ImageSetError('单张图片不能超过 25MB')
    ext = _ext_of(original)
    frame = decode_bgr(raw)
    if frame is None:
        raise ImageSetError('无法识别为图片，或尺寸不在 16 到 8000 像素之间')
    return {
        'original': original,
        'name': uuid.uuid4().hex[:16] + ext,
        'raw': raw,
    }


def _ext_of(original):
    ext = os.path.splitext(original or '')[1].lower()
    if ext == '.jpeg':
        ext = '.jpg'
    if ext not in _EXT_MIME:
        raise ImageSetError('只支持 png、jpg、bmp、webp')
    return ext


def _display_name(filename):
    base = os.path.basename(str(filename or '').replace('\\', '/')).replace('\x00', '').strip()
    return (base or 'image')[:80]


def _require_stored_name(name):
    if not _STORED_RE.fullmatch(name or ''):
        raise ImageSetError('图片不存在', 404)


def _mime(name):
    return _EXT_MIME.get(os.path.splitext(name)[1].lower(), 'application/octet-stream')


def _set_root(set_id):
    return os.path.join(current_app.config['IMAGE_FOLDER'], 'image-sets', set_id)


def _files_dir(set_id):
    return os.path.join(_set_root(set_id), 'files')


def _manifest_path(set_id):
    return os.path.join(_set_root(set_id), 'manifest.json')


def _stored_path(set_id, name):
    return os.path.join(_files_dir(set_id), name)


def _empty():
    return {'revision': 0, 'current': None, 'images': []}


def _load(set_id):
    path = _manifest_path(set_id)
    if not os.path.isfile(path):
        return _empty()
    try:
        with open(path, encoding='utf-8') as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ImageSetError(f'图片清单损坏: {exc}', 500)
    if not isinstance(data, dict):
        return _empty()
    images = data.get('images')
    if not isinstance(images, list):
        images = []
    clean = []
    for item in images:
        if not isinstance(item, dict):
            continue
        name = item.get('name')
        if not _STORED_RE.fullmatch(name or ''):
            continue
        label = item.get('label') or ''
        if label not in ('', 'normal', 'abnormal'):
            label = ''
        clean.append({
            'name': name,
            'label': label,
            'original': _display_name(item.get('original') or name),
        })
    current = data.get('current')
    if current and not any(item['name'] == current for item in clean):
        current = None
    return {
        'revision': int(data.get('revision') or 0),
        'current': current,
        'images': clean,
    }


def _save(set_id, data):
    folder = _set_root(set_id)
    os.makedirs(os.path.join(folder, 'files'), exist_ok=True)
    path = _manifest_path(set_id)
    tmp = path + '.tmp'
    payload = {
        'revision': int(data.get('revision') or 0),
        'current': data.get('current'),
        'images': data.get('images') or [],
    }
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _write_bytes(set_id, name, raw):
    folder = _files_dir(set_id)
    os.makedirs(folder, exist_ok=True)
    path = _stored_path(set_id, name)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as handle:
        handle.write(raw)
    os.replace(tmp, path)


def _mutate(set_id, fn):
    validate_set_id(set_id)
    with _LOCK:
        data = _load(set_id)
        fn(data)
        data['revision'] = int(data.get('revision') or 0) + 1
        _save(set_id, data)
        _JPEG_CACHE.clear()
        return _public(set_id, data)


def _public(set_id, data):
    return {
        'id': set_id,
        'revision': int(data.get('revision') or 0),
        'current': data.get('current') or None,
        'images': list(data.get('images') or []),
    }
