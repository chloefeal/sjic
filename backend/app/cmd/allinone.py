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
    target_fps = 20
    frame_delay = 1.0 / target_fps
    batch_size = 8
    batch = []
    
    # FPS 计算相关变量
    frame_count = 0
    batch_count = 0
    start_time = time.time()
    fps_update_interval = 2.0  # 每2秒更新一次FPS

    while cap.isOpened():
        batch_start_time = time.time()
        
        # 收集一个批次的帧
        while len(batch) < batch_size:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 预处理
            processed = preprocess(frame, new_h, new_w, top, bottom, left, right)
            if processed is not None:
                batch.append(processed)
                frame_count += 1
            
            # 控制采集帧率
            elapsed = time.time() - batch_start_time
            if elapsed < frame_delay * len(batch):
                time.sleep(frame_delay * len(batch) - elapsed)

        if not batch:
            break

        # 批量推理
        batch_tensor = torch.cat(batch)
        results = model(batch_tensor, imgsz=640, verbose=False)
        batch_count += 1
        
        # 显示结果
        for result in results:
            annotated_frame = result.plot()
            cv2.imshow("YOLO RTSP", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # 计算和显示FPS
        current_time = time.time()
        elapsed_total = current_time - start_time
        if elapsed_total >= fps_update_interval:
            processing_fps = frame_count / elapsed_total
            print(f"处理帧率: {processing_fps:.2f} FPS (frames: {frame_count}, batches: {batch_count})")
            # 重置计数器
            start_time = current_time
            frame_count = 0
            batch_count = 0

        # 清空批次
        batch = []

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    url = sys.argv[1]
    main(url)

