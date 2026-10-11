"""姿态场景的计时和冷却只在 PoseBehaviorAlgorithm.evaluate_behaviors 里做一次。"""
import sys
import types
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

# 判定测试不画图。缺 cv2 / numpy / shapely 时用空模块把包导入撑过去。
if 'cv2' not in sys.modules:
    sys.modules['cv2'] = types.ModuleType('cv2')
if 'numpy' not in sys.modules:
    numpy_mod = types.ModuleType('numpy')
    numpy_mod.ndarray = type('ndarray', (), {})
    sys.modules['numpy'] = numpy_mod
if 'shapely.geometry' not in sys.modules:
    shapely_mod = types.ModuleType('shapely')
    geometry_mod = types.ModuleType('shapely.geometry')

    class _Point:
        def __init__(self, xy):
            self.xy = xy

    class _Polygon:
        def __init__(self, points):
            self.points = points

        def contains(self, _point):
            return True

    geometry_mod.Point = _Point
    geometry_mod.Polygon = _Polygon
    geometry_mod.LineString = _Polygon
    shapely_mod.geometry = geometry_mod
    sys.modules['shapely'] = shapely_mod
    sys.modules['shapely.geometry'] = geometry_mod

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.pose_behavior import PoseBehaviorAlgorithm


class _Box:
    def __init__(self, confidence=0.9):
        self.confidence = confidence


class _Log:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)

    def info(self, *_args, **_kwargs):
        return None

    def debug(self, *_args, **_kwargs):
        return None


def _eval(algo, behaviors, persons, now, state, alerts, pose_only=False, algorithm_id=7):
    algo.evaluate_behaviors(
        behaviors,
        persons,
        now,
        'frame',
        'front',
        0.5,
        None,
        None,
        10,
        'task-1',
        state['since'],
        state['last'],
        state['extra'],
        {},
        {},
        lambda **kwargs: alerts.append(kwargs),
        _Log() if 'log' not in state else state['log'],
        algorithm_id=algorithm_id,
        pose_only=pose_only,
    )


class PoseSceneEvalTest(unittest.TestCase):
    def setUp(self):
        self.algo = PoseBehaviorAlgorithm()
        self.algo.draw_alert_overlay = lambda *args, **kwargs: 'overlay'
        self.spec = {
            'id': 'gaze',
            'type': 'gaze_away',
            'name': '张望',
            'seconds': 3,
            'alert_type': 'gaze_away',
        }
        self.box = _Box()
        self.persons = [(self.box, [[0, 0, 1]])]
        self.state = {
            'since': {'gaze': None},
            'last': {'gaze': None},
            'extra': {'gaze': {}},
        }
        self.alerts = []
        self.start = datetime(2026, 1, 1, 8, 0, 0)

    def test_first_hit_starts_timer_then_alerts_once_per_cooldown(self):
        with patch('algorithms.pose_behavior._eval_behavior', return_value=True):
            _eval(self.algo, [self.spec], self.persons, self.start, self.state, self.alerts)
            self.assertEqual(self.alerts, [])
            self.assertEqual(self.state['since']['gaze'], self.start)

            _eval(
                self.algo, [self.spec], self.persons,
                self.start + timedelta(seconds=2), self.state, self.alerts,
            )
            self.assertEqual(self.alerts, [])

            _eval(
                self.algo, [self.spec], self.persons,
                self.start + timedelta(seconds=3), self.state, self.alerts,
            )
            self.assertEqual(len(self.alerts), 1)
            self.assertEqual(self.alerts[0]['alert_type'], 'gaze_away')
            self.assertEqual(self.alerts[0]['algorithm_id'], 7)
            self.assertIn('持续 3.0s', self.alerts[0]['message'])

            _eval(
                self.algo, [self.spec], self.persons,
                self.start + timedelta(seconds=12), self.state, self.alerts,
            )
            self.assertEqual(len(self.alerts), 1)

            _eval(
                self.algo, [self.spec], self.persons,
                self.start + timedelta(seconds=13), self.state, self.alerts,
            )
        self.assertEqual(len(self.alerts), 2)

    def test_miss_resets_timer(self):
        with patch('algorithms.pose_behavior._eval_behavior', side_effect=[True, False, True]):
            _eval(self.algo, [self.spec], self.persons, self.start, self.state, self.alerts)
            _eval(
                self.algo, [self.spec], self.persons,
                self.start + timedelta(seconds=1), self.state, self.alerts,
            )
            self.assertIsNone(self.state['since']['gaze'])
            _eval(
                self.algo, [self.spec], self.persons,
                self.start + timedelta(seconds=4), self.state, self.alerts,
            )
        self.assertEqual(self.alerts, [])
        self.assertEqual(self.state['since']['gaze'], self.start + timedelta(seconds=4))

    def test_uses_only_the_first_matching_person(self):
        seen = []

        def fake(btype, kps, box, spec, view):
            seen.append(kps)
            return kps == 'first'

        second = (_Box(), 'second')
        with patch('algorithms.pose_behavior._eval_behavior', side_effect=fake):
            _eval(
                self.algo, [self.spec],
                [(self.box, 'first'), second],
                self.start, self.state, self.alerts,
            )
        self.assertEqual(seen, ['first'])

    def test_pose_only_skips_watch_and_mixed_task_ignores_it(self):
        watch = {'id': 'watch', 'type': 'smart_watch', 'name': '手表'}
        state = {
            'since': {'watch': None},
            'last': {'watch': None},
            'extra': {'watch': {}},
            'log': _Log(),
        }
        _eval(self.algo, [watch], [], self.start, state, self.alerts, pose_only=True)
        _eval(self.algo, [watch], [], self.start, state, self.alerts, pose_only=True)
        self.assertEqual(len(state['log'].warnings), 1)
        self.assertEqual(self.alerts, [])

        quiet = {
            'since': {'watch': None},
            'last': {'watch': None},
            'extra': {'watch': {}},
            'log': _Log(),
        }
        _eval(self.algo, [watch], [], self.start, quiet, self.alerts, pose_only=False)
        self.assertEqual(quiet['log'].warnings, [])
