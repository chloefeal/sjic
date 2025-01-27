from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    @abstractmethod
    def process(self, frame, parameters):
        """处理单帧图像"""
        pass

class ObjectDetectionAlgorithm(BaseAlgorithm):
    def process(self, frame, parameters):
        """目标检测算法"""
        model = parameters.get('model')
        confidence = parameters.get('confidence', 0.5)
        results = model(frame, conf=confidence)
        return results

class BehaviorAnalysisAlgorithm(BaseAlgorithm):
    def process(self, frame, parameters):
        """行为分析算法"""
        # 实现行为分析逻辑
        pass

class AlgorithmFactory:
    _algorithms = {
        'object_detection': ObjectDetectionAlgorithm,
        'behavior_analysis': BehaviorAnalysisAlgorithm
    }
    
    @classmethod
    def get_algorithm(cls, algorithm_type):
        """获取算法实例"""
        algorithm_class = cls._algorithms.get(algorithm_type)
        if not algorithm_class:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
        return algorithm_class()

    @classmethod
    def register_algorithm(cls, type_name, algorithm_class):
        """注册新算法"""
        cls._algorithms[type_name] = algorithm_class 