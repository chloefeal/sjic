import cv2
import numpy as np
from ultralytics import YOLO
import sys

def preprocess(frame, target_size=640):
    # 获取原始尺寸
    h, w = frame.shape[:2]
    
    # 计算缩放比例
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize 并添加灰边
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)  # 填充灰色 (114,114,114)
    padded[:new_h, :new_w] = resized
    
    # 归一化并转换通道顺序 (HWC -> CHW, BGR -> RGB)
    padded = padded.astype(np.float32) / 255.0  # 归一化到 [0,1]
    padded = padded.transpose(2, 0, 1)          # [H,W,C] -> [C,H,W]
    padded = np.ascontiguousarray(padded)       # 确保内存连续
    
    return padded

def main(url):
    # 加载模型
    model = YOLO("yolov8n.pt")  # 或加载 ONNX/TensorRT 模型

    # 打开 RTSP 流
    rtsp_url = url
    cap = cv2.VideoCapture(rtsp_url)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 预处理（Letterbox）
        processed = preprocess(frame, target_size=640)
        
        # 推理
        results = model(processed, imgsz=640, verbose=False)  # 指定输入尺寸
        
        # 后处理（绘制结果）
        annotated_frame = results[0].plot()  # 自动还原到原始尺寸
        
        # 显示
        cv2.imshow("YOLO RTSP", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    url = sys.argv[1]
    main(url)

