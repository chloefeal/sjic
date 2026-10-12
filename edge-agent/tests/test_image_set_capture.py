import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import utils.image_set as image_set  # noqa: E402


class _Resp:
    def __init__(self, status, content=b'', payload=None):
        self.status_code = status
        self.content = content
        self._payload = payload

    def json(self):
        return self._payload


def _jpeg(color):
    import cv2
    import numpy as np
    image = np.zeros((32, 40, 3), dtype=np.uint8)
    image[:, :] = color
    ok, buf = cv2.imencode('.jpg', image)
    if not ok:
        raise RuntimeError('encode failed')
    return buf.tobytes()


class TestImageSetCapture(unittest.TestCase):
    def setUp(self):
        self._saved = image_set._api_base
        image_set._api_base = 'http://platform.test/api/edge'

    def tearDown(self):
        image_set._api_base = self._saved

    def test_loops_current_image_and_switches(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest('cv2')
        first = _jpeg((0, 255, 0))
        second = _jpeg((255, 0, 0))
        state = {'name': 'aaaa.jpg', 'body': first}
        calls = {'image': 0}

        def fake_get(url, timeout=None):
            if url.endswith('/current'):
                calls['image'] += 1
                return _Resp(200, state['body'])
            return _Resp(200, b'', {'current': state['name'], 'revision': 1})

        original = image_set.requests.get
        image_set.requests.get = fake_get
        try:
            cap = image_set.ImageSetCapture('image-set:default')
            cap._frame_interval = 0
            cap._poll_interval = 0
            cap.start()
            ok, frame = cap.read()
            self.assertTrue(ok)
            self.assertEqual(tuple(frame[0, 0]), (0, 255, 0))
            self.assertTrue(cap.isOpened())
            pulled = calls['image']

            state['name'] = 'bbbb.jpg'
            state['body'] = second
            ok, frame = cap.read()
            self.assertTrue(ok)
            self.assertEqual(tuple(frame[0, 0]), (255, 0, 0))
            self.assertGreater(calls['image'], pulled)
            cap.release()
            self.assertFalse(cap.grab())
        finally:
            image_set.requests.get = original

    def test_open_rtsp_capture_routes_image_set(self):
        import utils.stream as stream
        sentinel = object()
        original = image_set.open_image_set_capture
        image_set.open_image_set_capture = lambda url: sentinel
        try:
            self.assertIs(stream.open_rtsp_capture('image-set:default'), sentinel)
        finally:
            image_set.open_image_set_capture = original
