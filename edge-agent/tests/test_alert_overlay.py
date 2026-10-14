"""告警截图默认不画骨骼，打开开关后才画点和连线。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.base import BaseAlgorithm, set_show_pose_skeleton, show_pose_skeleton


class _Algo(BaseAlgorithm):
    def process(self, camera_stream, config_dict, logger, stop_event, on_alert, runtime):
        return None


def _shoulders():
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    kps[5] = [8.0, 40.0, 1.0]
    kps[6] = [56.0, 40.0, 1.0]
    return kps


class TestAlertSkeletonOverlay(unittest.TestCase):
    def tearDown(self):
        set_show_pose_skeleton(False)

    def test_hidden_by_default(self):
        self.assertFalse(show_pose_skeleton())
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        out = _Algo().draw_alert_overlay(frame, skeletons=[_shoulders()])
        self.assertTrue(np.array_equal(out, frame))

    def test_string_false_stays_hidden(self):
        set_show_pose_skeleton('false')
        self.assertFalse(show_pose_skeleton())

    def test_draws_when_enabled(self):
        set_show_pose_skeleton(True)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        out = _Algo().draw_alert_overlay(frame, skeletons=[_shoulders()])
        self.assertTrue(np.any(out[40] != 0))

    def test_boxes_still_draw_when_skeleton_hidden(self):
        set_show_pose_skeleton(False)

        class _Box:
            x1, y1, x2, y2 = 4, 4, 20, 18
            class_id = 0
            confidence = 0.9

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        out = _Algo().draw_alert_overlay(frame, boxes=[_Box()], skeletons=[_shoulders()])
        self.assertTrue(np.all(out[40] == 0))
        self.assertGreater(int(np.sum(out[:, :, 1] > 200)), 0)


if __name__ == '__main__':
    unittest.main()
