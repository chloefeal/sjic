"""人员缺席/滞留/聚集：姿态框都算人，不看类别，也不看关键点。"""
import sys
import types
import unittest
from datetime import datetime, timedelta
from pathlib import Path

# 本机解释器未必装了 shapely；规则在无 ROI 时不会用到几何对象。
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
    shapely_mod.geometry = geometry_mod
    sys.modules['shapely'] = shapely_mod
    sys.modules['shapely.geometry'] = geometry_mod

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'algorithms'))

from rules import (  # noqa: E402
    AbsenceRule,
    CrowdCountRule,
    LingerRule,
    behavior_rois,
    filter_persons_roi,
    person_count_message,
    person_count_specs,
)


class _Box:
    def __init__(self, class_id=7, confidence=0.9, foot=(10, 20)):
        self.class_id = class_id
        self.confidence = confidence
        self.foot_center = foot


class _Log:
    def info(self, *_args, **_kwargs):
        return None


class PersonCountSpecTest(unittest.TestCase):
    def test_only_person_count_types_and_class_cleared(self):
        specs = person_count_specs([
            {'id': 'a', 'type': 'absence', 'class_ids': [2], 'absent_seconds': 10},
            {'id': 'gaze', 'type': 'gaze_away', 'seconds': 3},
            {'id': 'c', 'type': 'crowd_count', 'class_ids': [0], 'min_count': 2},
        ])
        self.assertEqual([item['id'] for item in specs], ['a', 'c'])
        self.assertEqual(specs[0]['class_ids'], [])
        self.assertEqual(specs[1]['class_ids'], [])

    def test_absence_linger_crowd_count_any_pose_box(self):
        log = _Log()
        start = datetime(2026, 1, 1, 8, 0, 0)
        person = _Box(class_id=4)

        absence = AbsenceRule(
            {'id': 'abs', 'type': 'absence', 'name': '人员缺席', 'class_ids': [], 'absent_seconds': 10, 'alert_type': 'person_absence'},
            None, 0,
        )
        self.assertIsNone(absence.evaluate([person], start, log))
        self.assertIsNone(absence.evaluate([], start + timedelta(seconds=9), log))
        hit = absence.evaluate([], start + timedelta(seconds=20), log)
        self.assertEqual(hit['alert_type'], 'person_absence')
        self.assertEqual(hit['boxes'], [])
        self.assertIn('持续无人', person_count_message({'name': '人员缺席'}, 'absence', hit))

        linger = LingerRule(
            {'id': 'lng', 'type': 'linger', 'name': '人员滞留', 'class_ids': [], 'linger_seconds': 5, 'alert_type': 'exam_area_linger'},
            None, 0,
        )
        self.assertIsNone(linger.evaluate([], start, log))
        self.assertIsNone(linger.evaluate([person], start, log))
        hit = linger.evaluate([person], start + timedelta(seconds=5), log)
        self.assertEqual(hit['boxes'], [person])
        self.assertIn('持续有人', person_count_message({'name': '人员滞留'}, 'linger', hit))

        crowd = CrowdCountRule(
            {'id': 'crd', 'type': 'crowd_count', 'name': '通道人员聚集', 'class_ids': [], 'min_count': 2, 'seconds': 3, 'alert_type': 'aisle_crowd'},
            None, 0,
        )
        boxes = [_Box(class_id=1), _Box(class_id=9)]
        self.assertIsNone(crowd.evaluate(boxes[:1], start, log))
        self.assertIsNone(crowd.evaluate(boxes, start, log))
        hit = crowd.evaluate(boxes, start + timedelta(seconds=3), log)
        self.assertEqual(len(hit['boxes']), 2)
        self.assertIn('人数 2', person_count_message({'name': '通道人员聚集', 'min_count': 2}, 'crowd_count', hit))

    def test_each_pose_scene_roi_defaults_to_full_frame(self):
        persons = [(_Box(foot=(1, 1)), 'kps'), (_Box(foot=(50, 50)), None)]
        self.assertEqual(filter_persons_roi(persons, None), persons)

        class _OnlyNearOrigin:
            def contains(self, point):
                return point.xy[0] < 10

        kept = filter_persons_roi(persons, _OnlyNearOrigin())
        self.assertEqual(kept, [persons[0]])

        rois = behavior_rois(
            [
                {'id': 'gaze', 'type': 'gaze_away'},
                {'id': 'aside', 'type': 'look_aside', 'detection_region': {'points': [{}, {}, {}]}},
            ],
            lambda region: [(0, 0), (10, 0), (0, 10)] if region else None,
        )
        self.assertEqual(rois['gaze'], (None, None))
        polygon, points = rois['aside']
        self.assertIsNotNone(polygon)
        self.assertEqual(len(points), 3)


if __name__ == '__main__':
    unittest.main()
