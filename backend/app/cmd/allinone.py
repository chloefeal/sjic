import cv2
import numpy as np
from ultralytics import YOLO
import sys, time
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

    # 目标帧率
    target_fps = 20
    frame_delay = 1.0 / target_fps  # 每帧的延迟时间

    # 初始化计数器和时间戳
    frame_count = 0
    start_time = time.time()

    batch_size = 8
    batch = []

    while cap.isOpened():
        # 记录当前帧的开始时间
        frame_start_time = time.time()
        # 增加帧计数
        frame_count += 1

        ret, frame = cap.read()
        if not ret:
            break
        
        # 预处理（Letterbox）
        processed = preprocess(frame, new_h, new_w, top, bottom, left, right)
        if processed is not None:
            batch.append(processed)
        else:
            continue

        if len(batch) == batch_size:
            # 合并批次并推理
            batch_tensor = torch.cat(batch)
            print("batch_tensor.shape:=================================", batch_tensor.shape)
            batch = []
        else:
            continue

        # 推理
        results = model(batch_tensor, imgsz=640, verbose=False)  # 指定输入尺寸
        print("results.lenght:=================================", len(results))

        #processed = frame
        #results = model(processed, verbose=False)  # 指定输入尺寸
        
        for result in results:
            # 后处理（绘制结果）
            annotated_frame = result.plot()  # 自动还原到原始尺寸
            
            # 显示
            cv2.imshow("YOLO RTSP", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # 计算当前帧的处理时间
            frame_elapsed_time = time.time() - frame_start_time
            # 计算需要延迟的时间
            sleep_time = frame_delay - frame_elapsed_time
            # 如果需要延迟，则进行延迟
            if sleep_time > 0:
                time.sleep(sleep_time)

            # 每隔一段时间计算一次处理帧率
            if frame_count % 30 == 0:  # 每30帧计算一次
                elapsed_time = time.time() - start_time
                processing_fps = frame_count / elapsed_time
                print(f"处理帧率: {processing_fps:.2f} FPS")
                start_time = time.time()
                frame_count = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    url = sys.argv[1]
    main(url)

