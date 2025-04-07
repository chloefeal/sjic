import sys
import os
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
from app.utils.calc import get_letterbox_params, preprocess

def main(url):
    # 加载模型
    model = YOLO("yolo11n.pt")  # 或加载 ONNX/TensorRT 模型

    # 打开 RTSP 流
    rtsp_url = url
    cap = cv2.VideoCapture(rtsp_url)

    h, w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    new_h, new_w, top, bottom, left, right = get_letterbox_params(h, w, target_size=640)
    if new_h is None:
        print(f"error: get_letterbox_params return None")
        return

    # 获取原始帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"原始帧率: {fps} FPS")

    # 目标帧率和批处理大小
    next_time = time.time()

    try:
        while cap.isOpened():
            # 持续跳过帧，直到到达下一秒
            while_times = 0
            while time.time() < next_time:
                if not cap.grab():  # 跳过帧（不解码）
                    print("无法获取帧或流已断开")
                    time.sleep(0.1)
                    continue
                while_times += 1
            print(f"while_times: {while_times}")
            next_time += 1
            ret, frame = cap.read()
            if not ret:
                break
                
            # 预处理
            processed = preprocess(frame, new_h, new_w, top, bottom, left, right)

            results = model(processed, imgsz=640, verbose=False)
            
            # 显示结果
            for result in results:
                annotated_frame = result.plot()
                cv2.imshow("YOLO RTSP", annotated_frame)
            
    except Exception as e:
        print(f"error: {e}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    url = sys.argv[1]
    main(url)

