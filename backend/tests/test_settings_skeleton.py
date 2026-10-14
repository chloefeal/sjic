import unittest

from app.models.setting import Setting


class TestSkeletonSetting(unittest.TestCase):
    def test_defaults_to_hidden(self):
        self.assertFalse(Setting.skeleton_overlay_enabled(None))
        self.assertFalse(Setting.skeleton_overlay_enabled({}))
        merged = Setting.merged_config({'alert': {'retention_days': 7}})
        self.assertFalse(merged['alert']['show_skeleton'])
        self.assertEqual(merged['alert']['retention_days'], 7)

    def test_explicit_values(self):
        self.assertTrue(Setting.skeleton_overlay_enabled({'alert': {'show_skeleton': True}}))
        self.assertFalse(Setting.skeleton_overlay_enabled({'alert': {'show_skeleton': False}}))
        self.assertTrue(Setting.skeleton_overlay_enabled({'alert': {'show_skeleton': 'true'}}))
        self.assertFalse(Setting.skeleton_overlay_enabled({'alert': {'show_skeleton': 'false'}}))
        self.assertFalse(Setting.skeleton_overlay_enabled({'alert': {'show_skeleton': 'no'}}))


if __name__ == '__main__':
    unittest.main()
