"""Synthetic COCO-17 keypoints covering exam-hall camera views."""
import unittest
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'algorithms'))

from pose_geometry import (
    accumulate_rising_count,
    is_chin_rest,
    is_cover_mouth,
    is_fall,
    is_finger_click_watch,
    is_gaze_away,
    is_head_down,
    is_looking_aside,
    is_standing,
    normalize_mount_position,
)

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16


def kps(**points):
    out = [[0.0, 0.0, 0.0] for _ in range(17)]
    names = {
        'nose': NOSE, 'l_eye': L_EYE, 'r_eye': R_EYE, 'l_ear': L_EAR, 'r_ear': R_EAR,
        'l_shoulder': L_SHOULDER, 'r_shoulder': R_SHOULDER,
        'l_elbow': L_ELBOW, 'r_elbow': R_ELBOW,
        'l_wrist': L_WRIST, 'r_wrist': R_WRIST,
        'l_hip': L_HIP, 'r_hip': R_HIP,
        'l_ankle': L_ANKLE, 'r_ankle': R_ANKLE,
    }
    for name, xyz in points.items():
        out[names[name]] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    return out


def sitting_torso(**extra):
    base = dict(
        l_shoulder=(100, 200, 0.9),
        r_shoulder=(180, 200, 0.9),
        l_hip=(110, 280, 0.85),
        r_hip=(170, 280, 0.85),
    )
    base.update(extra)
    return kps(**base)


class MountPositionTest(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_mount_position('front_top'), 'front_top')
        self.assertEqual(normalize_mount_position(''), 'back_top')
        self.assertEqual(normalize_mount_position('unknown'), 'back_top')


class BackTopExamSittingTest(unittest.TestCase):
    """现场图：后上方，考生低头看屏，只见后脑和双耳。"""

    def setUp(self):
        self.kps = sitting_torso(
            l_ear=(115, 155, 0.85),
            r_ear=(165, 155, 0.85),
        )

    def test_normal_look_at_screen_is_not_violation(self):
        self.assertFalse(is_looking_aside(self.kps, view='back_top'))
        self.assertFalse(is_head_down(self.kps, view='back_top'))
        self.assertFalse(is_gaze_away(self.kps, view='back_top'))
        self.assertFalse(is_fall(self.kps, view='back_top'))

    def test_peek_left_is_look_aside(self):
        peek = sitting_torso(
            l_ear=(80, 158, 0.9),
            r_ear=(140, 162, 0.25),
        )
        self.assertTrue(is_looking_aside(peek, view='back_top'))
        self.assertTrue(is_gaze_away(peek, view='back_top'))

    def test_turned_around_is_gaze_away_not_look_aside(self):
        face = sitting_torso(
            nose=(140, 170, 0.9),
            l_eye=(125, 165, 0.85),
            r_eye=(155, 165, 0.85),
            l_ear=(110, 168, 0.6),
            r_ear=(170, 168, 0.6),
        )
        self.assertFalse(is_looking_aside(face, view='back_top'))
        self.assertTrue(is_gaze_away(face, view='back_top'))


class FrontTopTest(unittest.TestCase):
    def test_looking_at_screen_is_not_gaze_away(self):
        """前上方看屏幕：鼻在肩下，这是正常考试，旧逻辑会误报。"""
        screen = sitting_torso(
            nose=(140, 210, 0.9),
            l_eye=(125, 200, 0.85),
            r_eye=(155, 200, 0.85),
            l_ear=(110, 185, 0.8),
            r_ear=(170, 185, 0.8),
        )
        self.assertFalse(is_looking_aside(screen, view='front_top'))
        self.assertFalse(is_head_down(screen, view='front_top'))
        self.assertFalse(is_gaze_away(screen, view='front_top'))

    def test_looking_at_camera_is_gaze_away(self):
        cam = sitting_torso(
            nose=(140, 150, 0.9),
            l_eye=(125, 140, 0.85),
            r_eye=(155, 140, 0.85),
            l_ear=(110, 155, 0.8),
            r_ear=(170, 155, 0.8),
        )
        self.assertTrue(is_gaze_away(cam, view='front_top'))

    def test_turn_right_is_look_aside(self):
        aside = sitting_torso(
            nose=(190, 195, 0.9),
            l_eye=(175, 185, 0.5),
            r_eye=(200, 185, 0.85),
            l_ear=(155, 185, 0.2),
            r_ear=(210, 185, 0.85),
        )
        self.assertTrue(is_looking_aside(aside, view='front_top'))


class SideTopTest(unittest.TestCase):
    def test_profile_looking_at_screen_is_not_aside(self):
        profile = sitting_torso(
            l_shoulder=(120, 200, 0.9),
            r_shoulder=(150, 205, 0.45),
            nose=(175, 175, 0.85),
            l_eye=(160, 168, 0.8),
            l_ear=(125, 165, 0.85),
        )
        self.assertFalse(is_looking_aside(profile, view='side_top'))
        self.assertFalse(is_gaze_away(profile, view='side_top'))

    def test_turned_toward_camera_is_aside(self):
        frontal = sitting_torso(
            nose=(140, 170, 0.9),
            l_eye=(125, 160, 0.85),
            r_eye=(155, 160, 0.85),
            l_ear=(110, 165, 0.7),
            r_ear=(170, 165, 0.7),
        )
        self.assertTrue(is_looking_aside(frontal, view='side_top'))


class TopViewTest(unittest.TestCase):
    def test_head_toward_desk_is_not_aside(self):
        desk = sitting_torso(
            l_ear=(115, 155, 0.8),
            r_ear=(165, 155, 0.8),
            nose=(140, 150, 0.4),
        )
        self.assertFalse(is_looking_aside(desk, view='top'))
        self.assertFalse(is_gaze_away(desk, view='top'))

    def test_head_along_shoulders_is_aside(self):
        aside = sitting_torso(
            l_ear=(165, 195, 0.8),
            r_ear=(200, 200, 0.7),
            nose=(190, 200, 0.35),
        )
        self.assertTrue(is_looking_aside(aside, view='top'))


class HandsAndFallTest(unittest.TestCase):
    def test_chin_rest(self):
        rest = sitting_torso(
            nose=(140, 170, 0.9),
            l_ear=(115, 165, 0.8),
            r_ear=(165, 165, 0.8),
            l_wrist=(138, 185, 0.9),
        )
        self.assertTrue(is_chin_rest(rest, view='front_top'))
        self.assertTrue(is_chin_rest(rest, view='back_top'))

    def test_cover_mouth(self):
        cover = sitting_torso(
            nose=(140, 170, 0.9),
            l_ear=(115, 165, 0.8),
            r_ear=(165, 165, 0.8),
            r_wrist=(142, 172, 0.9),
        )
        self.assertTrue(is_cover_mouth(cover, view='front_top'))
        self.assertTrue(is_cover_mouth(cover, view='back_top'))

    def test_sitting_is_not_fall(self):
        sit = sitting_torso(
            l_ear=(115, 155, 0.85),
            r_ear=(165, 155, 0.85),
        )
        self.assertFalse(is_fall(sit, view='back_top'))
        self.assertFalse(is_standing(sit, view='back_top'))

    def test_lying_is_fall(self):
        lie = kps(
            l_shoulder=(100, 200, 0.9),
            r_shoulder=(180, 205, 0.9),
            l_hip=(280, 210, 0.85),
            r_hip=(350, 215, 0.85),
            l_ear=(80, 198, 0.7),
            r_ear=(90, 202, 0.7),
        )
        self.assertTrue(is_fall(lie, view='back_top'))


class _Box:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2


class FingerClickWatchTest(unittest.TestCase):
    """肩宽 80。手表在左手腕，右手靠近才算点击。"""

    def setUp(self):
        self.watch = _Box(80, 240, 100, 260)

    def _person(self, right_wrist):
        return sitting_torso(
            l_wrist=(90, 250, 0.9),
            r_wrist=(*right_wrist, 0.9),
        )

    def test_other_hand_on_watch_is_click(self):
        self.assertTrue(is_finger_click_watch(self._person((125, 250)), [self.watch]))

    def test_wearing_without_other_hand_is_not_click(self):
        self.assertFalse(is_finger_click_watch(self._person((170, 260)), [self.watch]))

    def test_missing_wrist_is_not_click(self):
        kps_one_hand = sitting_torso(l_wrist=(90, 250, 0.9))
        self.assertFalse(is_finger_click_watch(kps_one_hand, [self.watch]))

    def test_no_watch_box_is_not_click(self):
        self.assertFalse(is_finger_click_watch(self._person((125, 250)), []))


class RisingCountTest(unittest.TestCase):
    def test_one_click_triggers_immediately(self):
        state = {}
        now = datetime(2026, 10, 8, 9, 0, 0)
        self.assertTrue(accumulate_rising_count(state, True, now, 1, 30, hold_contact=True))
        self.assertFalse(accumulate_rising_count(
            state, True, now + timedelta(seconds=1), 1, 30, hold_contact=True
        ))
        self.assertFalse(accumulate_rising_count(
            state, False, now + timedelta(seconds=2), 1, 30, hold_contact=True
        ))
        self.assertTrue(accumulate_rising_count(
            state, True, now + timedelta(seconds=3), 1, 30, hold_contact=True
        ))

    def test_glasses_needs_three_separate_touches(self):
        state = {}
        now = datetime(2026, 10, 8, 9, 0, 0)
        self.assertFalse(accumulate_rising_count(state, True, now, 3, 30))
        self.assertFalse(accumulate_rising_count(state, True, now + timedelta(seconds=1), 3, 30))
        self.assertFalse(accumulate_rising_count(state, False, now + timedelta(seconds=2), 3, 30))
        self.assertFalse(accumulate_rising_count(state, True, now + timedelta(seconds=3), 3, 30))
        self.assertFalse(accumulate_rising_count(state, False, now + timedelta(seconds=4), 3, 30))
        self.assertTrue(accumulate_rising_count(state, True, now + timedelta(seconds=5), 3, 30))

    def test_window_resets_old_clicks(self):
        state = {}
        now = datetime(2026, 10, 8, 9, 0, 0)
        self.assertFalse(accumulate_rising_count(state, True, now, 2, 30, hold_contact=True))
        self.assertFalse(accumulate_rising_count(
            state, False, now + timedelta(seconds=31), 2, 30, hold_contact=True
        ))
        self.assertFalse(accumulate_rising_count(
            state, True, now + timedelta(seconds=32), 2, 30, hold_contact=True
        ))


if __name__ == '__main__':
    unittest.main()
