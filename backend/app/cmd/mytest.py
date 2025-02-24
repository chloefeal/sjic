import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.algorithms.base import BaseAlgorithm
from app.algorithms.object_detection import ObjectDetectionAlgorithm
from app.algorithms.template_algorithm import TemplateAlgorithm
from app.algorithms.belt_broken import BeltBroken
from app.algorithms.belt_deviation_detection import BeltDeviationDetection
import cv2
from app.models import Task
from app.models import DetectionModel
from config import Config
from ultralytics import YOLO
from app import app, db
from app.models import Camera

def main():
    algo = ObjectDetectionAlgorithm()
    task_id = 1

    app.app_context().push()

    task = Task.query.filter_by(id=task_id).first()

    camera = Camera.query.filter_by(id=task.cameraId).first()
    rtsp_url = camera.url
    cap = cv2.VideoCapture(rtsp_url)

    modelId = task.modelId
    model = DetectionModel.query.get(modelId)
    if not model:
        raise ValueError(f"Model with id {modelId} not found")
    # 使用绝对路径
    model_path = os.path.join(Config.MODEL_FOLDER, model.path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    source = "1.png"
    yolo_model = YOLO(model_path)
    parameters = {
                    'model': yolo_model,
                    'confidence': task.confidence,
                    'alertThreshold': task.alertThreshold,
                    'algorithm_parameters': task.algorithm_parameters,
                    'on_alert': None  # 传递告警处理回调
                }

    result = algo.process(cap, parameters)
    print("Process result:", result)



if __name__ == "__main__":
    main()