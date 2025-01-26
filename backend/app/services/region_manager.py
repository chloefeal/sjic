import cv2
import numpy as np
import json

class RegionManager:
    def __init__(self):
        self.regions = {}
        
    def add_region(self, region_id, points, region_type):
        """添加监控区域"""
        self.regions[region_id] = {
            'points': np.array(points),
            'type': region_type
        }
        
    def save_regions(self, filepath):
        """保存区域设置到文件"""
        regions_dict = {
            k: {'points': v['points'].tolist(), 'type': v['type']}
            for k, v in self.regions.items()
        }
        with open(filepath, 'w') as f:
            json.dump(regions_dict, f)
            
    def load_regions(self, filepath):
        """从文件加载区域设置"""
        with open(filepath, 'r') as f:
            regions_dict = json.load(f)
        for k, v in regions_dict.items():
            self.regions[k] = {
                'points': np.array(v['points']),
                'type': v['type']
            }
            
    def check_point_in_region(self, point, region_id):
        """检查点是否在指定区域内"""
        if region_id not in self.regions:
            return False
        return cv2.pointPolygonTest(self.regions[region_id]['points'], point, False) >= 0 