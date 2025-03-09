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
from app.utils.calc import get_letterbox_params

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
    
    # 检查视频流是否成功打开
    if not cap.isOpened():
        print(f"错误: 无法打开视频流 {url}")
        return

    # 获取原始帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"原始帧率: {fps} FPS, 尺寸: {w}x{h}")
    
    # 计算缩放参数
    target_size = 800  # Faster R-CNN 通常使用更大的输入尺寸
    new_h, new_w, top, bottom, left, right = get_letterbox_params(h, w, target_size=target_size)
    if new_h is None:
        print(f"错误: get_letterbox_params 返回 None")
        return
    
    print(f"缩放后尺寸: {new_w}x{new_h}, padding: top={top}, bottom={bottom}, left={left}, right={right}")
    
    # FPS 计算相关变量
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 2.0  # 每2秒更新一次FPS
    
    # 颜色映射（用于绘制不同类别的边界框）
    color_map = {}
    
    # 置信度阈值
    confidence_threshold = 0.5

    def preprocess_for_rcnn(image):
        """自定义预处理函数，不依赖于 app.utils.calc.preprocess"""
        try:
            # 计算缩放比例
            scale = min(target_size / h, target_size / w)
            
            # 计算新尺寸
            resize_h, resize_w = int(h * scale), int(w * scale)
            
            # 缩放图像
            resized = cv2.resize(image, (resize_w, resize_h))
            
            # 创建画布
            canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            
            # 计算偏移量以居中放置
            offset_x = (target_size - resize_w) // 2
            offset_y = (target_size - resize_h) // 2
            
            # 将缩放后的图像放在画布中央
            canvas[offset_y:offset_y+resize_h, offset_x:offset_x+resize_w] = resized
            
            # 转换为 RGB
            image_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            
            # 转换为 PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # 转换为张量
            tensor = F.to_tensor(pil_image)
            
            return tensor, canvas, (offset_x, offset_y, scale)
            
        except Exception as e:
            print(f"预处理错误: {e}")
            return None

    def draw_boxes(output_image, boxes, labels, scores, preprocess_info):
        """在图像上绘制检测框"""
        offset_x, offset_y, scale = preprocess_info
        
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
            
            # 获取边界框坐标
            x1, y1, x2, y2 = box
            
            # 绘制边界框
            cv2.rectangle(output_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # 绘制标签
            label_text = f"{class_name}: {score:.2f}"
            cv2.putText(output_image, label_text, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output_image

    # 创建窗口
    cv2.namedWindow("Faster R-CNN RTSP", cv2.WINDOW_NORMAL)
    cv2.namedWindow("原始视频", cv2.WINDOW_NORMAL)
    cv2.namedWindow("预处理后", cv2.WINDOW_NORMAL)
    
    print("开始处理视频流...")

    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧，退出...")
            break
        
        # 显示原始帧
        cv2.imshow("原始视频", frame)
        
        # 预处理
        result = preprocess_for_rcnn(frame)
        if result is None:
            print("预处理失败，跳过当前帧")
            continue
            
        tensor, processed_frame, preprocess_info = result
        
        # 显示处理后的帧
        cv2.imshow("预处理后", processed_frame)
        
        # 推理
        with torch.no_grad():
            input_tensor = tensor.unsqueeze(0).to(device)
            result = model(input_tensor)[0]
        
        # 获取检测结果
        boxes = result['boxes'].cpu().numpy()
        labels = result['labels'].cpu().numpy()
        scores = result['scores'].cpu().numpy()
        
        # 绘制检测结果
        annotated_frame = draw_boxes(processed_frame.copy(), boxes, labels, scores, preprocess_info)
        
        # 显示帧率
        current_time = time.time()
        elapsed_total = current_time - start_time
        frame_count += 1
        
        if elapsed_total > 0:
            current_fps = frame_count / elapsed_total
            cv2.putText(annotated_frame, f"FPS: {current_fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow("Faster R-CNN RTSP", annotated_frame)
        
        # 计算和显示FPS
        if elapsed_total >= fps_update_interval:
            processing_fps = frame_count / elapsed_total
            print(f"处理帧率: {processing_fps:.2f} FPS (frames: {frame_count})")
            # 重置计数器
            start_time = current_time
            frame_count = 0
        
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("处理完成")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python allinone.fasterrcnn.py <视频URL>")
        sys.exit(1)
    url = sys.argv[1]
    main(url)
