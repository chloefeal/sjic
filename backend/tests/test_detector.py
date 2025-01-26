import unittest
import cv2
import numpy as np
from app.services.detector import Detector

class TestDetector(unittest.TestCase):
    def setUp(self):
        self.detector = Detector()
        
    def test_belt_deviation_detection(self):
        # 创建测试图像
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.line(test_image, (100, 240), (540, 240), (255, 255, 255), 2)
        
        # 测试检测
        result = self.detector._detect_deviation(test_image)
        self.assertIsInstance(result, dict)
        self.assertIn('deviation', result)
        
    def test_material_detection(self):
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(test_image, (320, 240), 50, (255, 255, 255), -1)
        
        result = self.detector._detect_material(test_image)
        self.assertIsInstance(result, dict)
        self.assertIn('material_detected', result) 