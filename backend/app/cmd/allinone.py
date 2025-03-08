import cv2
import numpy as np
from ultralytics import YOLO
import sys, time
import torch


def preprocess(frame, target_size=640):
    # 1. 检查输入有效性
    if frame is None or frame.size == 0:
        return None

    # 获取原始尺寸
    print(frame.shape)
    h, w = frame.shape[:2]
    
    # 计算缩放比例
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    print(h, w)
    print(new_h, new_w)

    dh = target_size - new_h
    dw = target_size - new_w
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    print(dh, dw)
    print(top, bottom, left, right)
    
    # Resize 并添加灰边
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    padded = padded.transpose(2, 0, 1).astype(np.float32) / 255.0  # HWC -> CHW, 归一化
    padded = np.ascontiguousarray(padded)       # 确保内存连续
    # 转换为Tensor并添加批次维度
    padded = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).cuda()  # 转为张量并移至 GPU
    #padded = torch.tensor(padded).unsqueeze(0).cuda()  # 转为张量并移至 GPU

    print("预处理输出形状:", padded.shape)  # 应为 [1,3,640,640]
    print("数据类型:", padded.dtype)     # 应为 float32
    
    return padded

def main(url):
    # 加载模型
    model = YOLO("yolo11n.pt")  # 或加载 ONNX/TensorRT 模型

    # 打开 RTSP 流
    rtsp_url = url
    cap = cv2.VideoCapture(rtsp_url)
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

        ret, frame = cap.read()
        if not ret:
            break
        
        # 预处理（Letterbox）
        processed = preprocess(frame, target_size=640)
        if processed is not None:
            batch.append(processed)

        if len(batch) == batch_size:
            # 合并批次并推理
            batch_tensor = torch.cat(batch)

        # 推理
        results = model(processed, imgsz=640, verbose=False)  # 指定输入尺寸
        print("results.lenght:=================================", len(results))

        #processed = frame
        #results = model(processed, verbose=False)  # 指定输入尺寸
        
        # 后处理（绘制结果）
        annotated_frame = results[0].plot()  # 自动还原到原始尺寸
        
        # 显示
        cv2.imshow("YOLO RTSP", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # 增加帧计数
        frame_count += 1

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

        batch = []  # 清空批次	

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    url = sys.argv[1]
    main(url)

