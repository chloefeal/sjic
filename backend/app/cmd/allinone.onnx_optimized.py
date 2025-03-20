import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import cv2
import numpy as np
import time
import onnxruntime as ort
from app.utils.calc import get_letterbox_params, preprocess

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def nms(boxes, scores, iou_threshold=0.45):
    """非极大值抑制"""
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
            
        current_box = boxes[current]
        rest_boxes = boxes[indices[1:]]
        
        ious = np.array([calculate_iou(current_box, box) for box in rest_boxes])
        indices = indices[1:][ious < iou_threshold]
    
    return keep

def main(url, model_path):
    # 使用 ONNX Runtime 的会话选项减少内存使用
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 加载 ONNX 模型
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, options, providers=providers)
        print(f"已加载 ONNX 模型 (CUDA): {model_path}")
    except Exception as e:
        print(f"CUDA 加载失败: {str(e)}")
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(model_path, options, providers=providers)
        print(f"已加载 ONNX 模型 (CPU): {model_path}")
    
    # 获取模型信息
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"模型输入名称: {input_name}, 形状: {input_shape}")
    
    # 打开视频流
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"错误: 无法打开视频流 {url}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"原始帧率: {fps} FPS, 尺寸: {w}x{h}")
    
    # 计算缩放参数
    target_size = 640
    new_h, new_w, top, bottom, left, right = get_letterbox_params(h, w, target_size=target_size)
    
    # FPS 计算
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 2.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # 预处理
            resized = cv2.resize(frame, (new_w, new_h))
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            
            # 转换为模型输入格式
            input_tensor = padded.astype(np.float32) / 255.0
            input_tensor = input_tensor.transpose(2, 0, 1)  # HWC to CHW
            input_tensor = np.expand_dims(input_tensor, axis=0)  # 添加批次维度
            input_tensor = np.ascontiguousarray(input_tensor)  # 确保内存连续
            
            # 推理
            outputs = session.run(None, {input_name: input_tensor})
            
            # 处理输出 - 假设输出是 [batch, num_boxes, 85] 格式 (YOLO)
            if len(outputs) == 1 and len(outputs[0].shape) == 3:
                predictions = outputs[0][0]  # 取第一个批次
                
                # 过滤低置信度检测
                mask = predictions[:, 4] > 0.25
                filtered_preds = predictions[mask]
                
                if len(filtered_preds) > 0:
                    # 提取边界框、置信度和类别
                    boxes = []
                    scores = []
                    class_ids = []
                    
                    for pred in filtered_preds:
                        # 边界框 (中心点, 宽高)
                        cx, cy, w_box, h_box = pred[:4]
                        
                        # 转换为左上角和右下角坐标
                        x1 = int((cx - w_box/2) * w / target_size)
                        y1 = int((cy - h_box/2) * h / target_size)
                        x2 = int((cx + w_box/2) * w / target_size)
                        y2 = int((cy + h_box/2) * h / target_size)
                        
                        # 获取类别和分数
                        obj_conf = pred[4]
                        class_scores = pred[5:]
                        class_id = np.argmax(class_scores)
                        score = class_scores[class_id] * obj_conf
                        
                        if score > 0.25:
                            boxes.append([x1, y1, x2, y2])
                            scores.append(score)
                            class_ids.append(class_id)
                    
                    if boxes:
                        boxes = np.array(boxes)
                        scores = np.array(scores)
                        class_ids = np.array(class_ids)
                        
                        # 应用 NMS
                        keep = nms(boxes, scores)
                        boxes = boxes[keep]
                        scores = scores[keep]
                        class_ids = class_ids[keep]
                        
                        # 绘制检测结果
                        for box, score, cls_id in zip(boxes, scores, class_ids):
                            x1, y1, x2, y2 = box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"Class {int(cls_id)}: {score:.2f}", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 显示帧率
            current_time = time.time()
            elapsed = current_time - start_time
            frame_count += 1
            
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow("ONNX Detection", frame)
            
            # 更新 FPS
            if elapsed >= fps_update_interval:
                print(f"处理帧率: {frame_count/elapsed:.2f} FPS")
                start_time = current_time
                frame_count = 0
                
        except Exception as e:
            print(f"推理过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python allinone.onnx_optimized.py <视频URL> <ONNX模型文件路径>")
        sys.exit(1)
    url = sys.argv[1]
    model_path = sys.argv[2]
    main(url, model_path) 