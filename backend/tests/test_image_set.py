import os
import shutil
import tempfile
import unittest

from flask import Flask

from app.models.camera import Camera
from app.services.image_set import ImageSetError, add_images, camera_url, delete_image, set_current, snapshot


def _jpeg_bytes():
    import cv2
    import numpy as np
    image = np.zeros((32, 40, 3), dtype=np.uint8)
    image[:, :] = (0, 180, 255)
    ok, buf = cv2.imencode('.jpg', image)
    if not ok:
        raise RuntimeError('encode failed')
    return buf.tobytes()


class TestImageSet(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.app = Flask(__name__)
        self.app.config['IMAGE_FOLDER'] = self.tmp
        self.ctx = self.app.app_context()
        self.ctx.push()

    def tearDown(self):
        self.ctx.pop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_rtsp_url_keeps_image_set_marker(self):
        camera = Camera(name='图片测试', url='image-set:default')
        self.assertEqual(camera.get_rtsp_url(), 'image-set:default')
        video = Camera(name='文件', url='clip.mp4')
        self.assertEqual(os.path.basename(video.get_rtsp_url()), 'clip.mp4')
        self.assertNotEqual(video.get_rtsp_url(), 'clip.mp4')
        live = Camera(name='流', url='rtsp://example/live')
        self.assertEqual(live.get_rtsp_url(), 'rtsp://example/live')

    def test_upload_select_and_delete(self):
        try:
            raw = _jpeg_bytes()
        except ImportError:
            self.skipTest('cv2')
        added = add_images('default', [{'original': '现场.jpg', 'raw': raw}])
        self.assertEqual(len(added['images']), 1)
        name = added['images'][0]['name']
        self.assertEqual(added['images'][0]['original'], '现场.jpg')
        self.assertIsNone(added['current'])

        selected = set_current('default', name)
        self.assertEqual(selected['current'], name)
        self.assertEqual(snapshot('default')['current'], name)

        cleared = delete_image('default', name)
        self.assertEqual(cleared['images'], [])
        self.assertIsNone(cleared['current'])
        self.assertEqual(camera_url('default'), 'image-set:default')

    def test_rejects_path_and_bad_image(self):
        with self.assertRaises(ImageSetError):
            snapshot('../etc')
        with self.assertRaises(ImageSetError):
            add_images('default', [{'original': 'notes.txt', 'raw': b'hello'}])
        with self.assertRaises(ImageSetError):
            set_current('default', '../../etc/passwd')
