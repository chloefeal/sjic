import time
import unittest

from app.services.detection_preview import PreviewHub, normalize_labelmap, preview_engines


class _Task:
    def __init__(self, **kwargs):
        self.algorithm_id = kwargs.get('algorithm_id')
        self.od_algorithm_id = kwargs.get('od_algorithm_id')
        self.pose_algorithm_id = kwargs.get('pose_algorithm_id')

    def is_exam_pipeline(self):
        return bool(self.od_algorithm_id or self.pose_algorithm_id)


class TestPreviewMetadata(unittest.TestCase):
    def test_labelmap_array_and_object(self):
        self.assertEqual(
            normalize_labelmap([{'id': 0, 'name': '手机'}, {'id': '2', 'name': '书'}]),
            {0: '手机', 2: '书'},
        )
        self.assertEqual(normalize_labelmap({'1': '帽子'}), {1: '帽子'})

    def test_engines(self):
        exam = _Task(od_algorithm_id=3, pose_algorithm_id=4)
        self.assertEqual(preview_engines(exam), ['object_detection', 'pose_behavior'])
        pose_only = _Task(pose_algorithm_id=4)
        self.assertEqual(preview_engines(pose_only), ['pose_behavior'])
        single = _Task(algorithm_id=9)
        self.assertEqual(preview_engines(single, engine='object_detection'), ['object_detection'])
        self.assertEqual(preview_engines(single, engine='camera_health'), [])


class TestPreviewHub(unittest.TestCase):
    def test_ingest_follows_the_open_view(self):
        hub = PreviewHub()
        self.assertFalse(hub.ingest(1, b'\xff\xd8', {'object': {'count': 1}}))
        self.assertEqual(hub.set_view(1, 'a', True, 10), 'on')
        self.assertTrue(hub.ingest(1, b'\xff\xd8ok', {'object': {'count': 4}}))
        snap = hub.snapshot(1)
        self.assertFalse(snap['waiting'])
        self.assertEqual(snap['stats']['object']['count'], 4)
        self.assertTrue(snap['image'])

        self.assertEqual(hub.set_view(1, 'a', True, 9), 'ignored')
        self.assertEqual(hub.set_view(1, 'a', False, 11), 'off')
        self.assertTrue(hub.snapshot(1)['waiting'])
        self.assertFalse(hub.ingest(1, b'\xff\xd8', {'object': {'count': 9}}))

    def test_second_viewer_keeps_the_stream(self):
        hub = PreviewHub()
        hub.set_view(1, 'a', True, 1)
        hub.set_view(1, 'b', True, 2)
        self.assertEqual(hub.set_view(1, 'a', False, 3), 'extend')
        self.assertTrue(hub.is_live(1))
        self.assertEqual(hub.set_view(1, 'b', False, 4), 'off')
        self.assertFalse(hub.is_live(1))

    def test_expired_view_drops_the_frame(self):
        hub = PreviewHub()
        hub.set_view(1, 'a', True, 1, ttl=0.01)
        self.assertTrue(hub.ingest(1, b'\xff\xd8', {'pose': {'count': 2}}))
        time.sleep(0.02)
        snap = hub.snapshot(1)
        self.assertTrue(snap['waiting'])
        self.assertIsNone(snap['image'])


if __name__ == '__main__':
    unittest.main()
