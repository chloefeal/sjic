from .base import BaseAlgorithm
from .object_detection import ObjectDetectionAlgorithm
from .template_algorithm import TemplateAlgorithm
from .belt_broken import BeltBroken
from .belt_deviation_detection import BeltDeviationDetection

__all__ = [
    'BaseAlgorithm',
    'ObjectDetectionAlgorithm',
    'TemplateAlgorithm',
    'BeltBroken',
    'BeltDeviationDetection'
]