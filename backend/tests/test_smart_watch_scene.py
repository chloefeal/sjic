import importlib.util
import unittest
from pathlib import Path


def _load(name):
    path = Path(__file__).resolve().parents[1] / 'app' / 'utils' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


catalog = _load('algorithm_catalog')
exam_task = _load('exam_task')


class SmartWatchSceneTest(unittest.TestCase):
    def test_catalog_defaults_to_one_click(self):
        preset = next(item for item in catalog.FUSION_SCENE_PRESETS if item['id'] == 'smart_watch')
        self.assertEqual(preset['name'], '智能手表作弊检测')
        self.assertEqual(preset['type'], 'smart_watch')
        self.assertEqual(preset['requires'], ['object_detection', 'pose_behavior'])
        self.assertEqual(preset['defaults']['min_touches'], 1)
        self.assertEqual(preset['defaults']['alert_type'], 'smart_watch')
        self.assertEqual(catalog.alert_type_labels()['smart_watch'], '智能手表作弊检测')

    def test_pose_schema_does_not_keep_fusion_scenes(self):
        merged = catalog.merge_catalog_schema(
            {'scene_presets': [
                {'id': 'smart_watch', 'type': 'smart_watch', 'name': '智能手表作弊检测'},
                {'id': 'look_aside', 'type': 'look_aside', 'name': '左右张望'},
            ]},
            'pose_behavior',
        )
        ids = [item['id'] for item in merged['scene_presets']]
        self.assertNotIn('smart_watch', ids)
        self.assertNotIn('smart_glasses', ids)
        self.assertIn('look_aside', ids)

    def test_watch_fusion_needs_both_models_and_class(self):
        params = {'fusions': [{'type': 'smart_watch', 'class_ids': [4], 'enabled': True}]}
        self.assertEqual(
            exam_task.validate_scene_bindings(params, None, 2),
            '智能手表需要同时绑定目标检测算法和姿态算法',
        )
        self.assertEqual(
            exam_task.validate_scene_bindings({'fusions': [{'type': 'smart_watch', 'enabled': True}]}, 1, 2),
            '智能手表场景需要选择手表类别',
        )
        self.assertIsNone(exam_task.validate_scene_bindings(params, 1, 2))

    def test_glasses_class_error_stays_specific(self):
        self.assertEqual(
            exam_task.validate_scene_bindings(
                {'fusions': [{'type': 'smart_glasses', 'enabled': True}]},
                1,
                2,
            ),
            '智能眼镜场景需要选择眼镜类别',
        )

    def test_watch_is_removed_from_pose_behaviors(self):
        params = exam_task.sanitize_exam_params({
            'behaviors': [
                {'type': 'smart_watch', 'name': '手表'},
                {'type': 'look_aside', 'name': '张望'},
            ],
            'fusions': [{'type': 'smart_watch', 'class_ids': [4], 'enabled': True}],
        })
        self.assertEqual([item['type'] for item in params['behaviors']], ['look_aside'])
        needs = exam_task.scene_needs(params)
        self.assertTrue(needs['need_od'])
        self.assertTrue(needs['need_pose'])
        self.assertEqual(needs['fusions'][0]['type'], 'smart_watch')


if __name__ == '__main__':
    unittest.main()
