import sys
import os
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import cv2
import numpy as np
import time
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
from app.utils.calc import get_letterbox_params, preprocess

def main(url):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载 Faster R-CNN 模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()

    # COCO 数据集的类别
    COCO_CLASSES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # 打开视频流
    rtsp_url = url
    cap = cv2.VideoCapture(rtsp_url)

    # 获取原始帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"原始帧率: {fps} FPS")

    # 目标帧率和批处理大小
    target_fps = 10  # Faster R-CNN 比 YOLO 慢，降低目标帧率
    frame_delay = 1.0 / target_fps
    batch_size = 4  # 减小批处理大小
    
    # FPS 计算相关变量
    frame_count = 0
    batch_count = 0
    start_time = time.time()
    fps_update_interval = 2.0  # 每2秒更新一次FPS
    
    # 颜色映射（用于绘制不同类别的边界框）
    color_map = {}
    
    # 置信度阈值
    confidence_threshold = 0.5

    def preprocess_for_rcnn(image):
        # 转换为 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 转换为 PIL Image
        pil_image = Image.fromarray(image_rgb)
        # 转换为张量
        tensor = F.to_tensor(pil_image)
        return tensor

    def draw_boxes(image, boxes, labels, scores):
        # 绘制检测结果
        for box, label, score in zip(boxes, labels, scores):
            if score < confidence_threshold:
                continue
                
            # 获取类别名称
            class_name = COCO_CLASSES[label]
            
            # 为每个类别分配一个固定的颜色
            if class_name not in color_map:
                color_map[class_name] = tuple(np.random.randint(0, 255, 3).tolist())
            color = color_map[class_name]
            
            # 绘制边界框
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # 绘制标签
            label_text = f"{class_name}: {score:.2f}"
            cv2.putText(image, label_text, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image

    while cap.isOpened():
        batch_start_time = time.time()
        batch_frames = []  # 原始帧
        batch_tensors = []  # 处理后的张量
        
        # 收集一个批次的帧
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
                
            # 保存原始帧用于显示
            batch_frames.append(frame.copy())
            
            # 预处理
            tensor = preprocess_for_rcnn(frame)
            batch_tensors.append(tensor)
            frame_count += 1
            
            # 控制采集帧率
            elapsed = time.time() - batch_start_time
            if elapsed < frame_delay * len(batch_tensors):
                time.sleep(frame_delay - (elapsed % frame_delay))
        
        if not batch_tensors:
            break
        
        # 批量推理
        with torch.no_grad():
            batch_results = []
            for tensor in batch_tensors:
                # Faster R-CNN 需要列表输入
                input_tensor = tensor.unsqueeze(0).to(device)
                result = model(input_tensor)[0]
                batch_results.append(result)
        
        batch_count += 1
        
        # 显示结果
        for frame, result in zip(batch_frames, batch_results):
            # 获取检测结果
            boxes = result['boxes'].cpu().numpy()
            labels = result['labels'].cpu().numpy()
            scores = result['scores'].cpu().numpy()
            
            # 绘制检测结果
            annotated_frame = draw_boxes(frame, boxes, labels, scores)
            
            # 显示帧率
            current_time = time.time()
            elapsed_total = current_time - start_time
            if elapsed_total > 0:
                current_fps = frame_count / elapsed_total
                cv2.putText(annotated_frame, f"FPS: {current_fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow("Faster R-CNN RTSP", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return
        
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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python allinone.fasterrcnn.py <视频URL>")
        sys.exit(1)
    url = sys.argv[1]
    main(url)
