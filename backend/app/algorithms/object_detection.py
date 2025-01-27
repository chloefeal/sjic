from .base import BaseAlgorithm

class ObjectDetectionAlgorithm(BaseAlgorithm):
    """目标检测算法"""
    __tablename__ = 'algorithms'  # 使用同一个表
    __mapper_args__ = {'polymorphic_identity': 'ObjectDetectionAlgorithm'}

    def process(self, frame, parameters):
        """目标检测算法"""
        model = parameters.get('model')
        confidence = parameters.get('confidence', 0.5)
        results = model(frame, conf=confidence)
        return results

    def get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.5
                },
                "regions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "width": {"type": "number"},
                            "height": {"type": "number"}
                        }
                    }
                }
            }
        }

    def validate_parameters(self, parameters):
        # TODO: 实现参数验证
        return True 