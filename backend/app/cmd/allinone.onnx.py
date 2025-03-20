import sys
import os
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import cv2
import numpy as np
import time
import torch
import onnxruntime as ort
from app.utils.calc import get_letterbox_params, preprocess

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    # 计算交集区域的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集区域的面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算两个边界框的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集的面积
    union = area1 + area2 - intersection
    
    # 计算IoU
    iou = intersection / union if union > 0 else 0
    
    return iou

def nms(boxes, scores, iou_threshold=0.45):
    """非极大值抑制的简单实现"""
    # 按照分数降序排序
    sorted_indices = np.argsort(scores)[::-1]
    
    keep = []
    while sorted_indices.size > 0:
        # 保留分数最高的框
        current_index = sorted_indices[0]
        keep.append(current_index)
        
        # 如果只剩一个框，结束循环
        if sorted_indices.size == 1:
            break
        
        # 计算当前框与其他框的IoU
        current_box = boxes[current_index]
        other_indices = sorted_indices[1:]
        other_boxes = boxes[other_indices]
        
        # 计算IoU
        ious = np.array([calculate_iou(current_box, other_box) for other_box in other_boxes])
        
        # 保留IoU小于阈值的框
        mask = ious < iou_threshold
        sorted_indices = other_indices[mask]
    
    return keep

def process_outputs(outputs, input_shape, orig_shape, conf_threshold=0.25, iou_threshold=0.45):
    """处理模型输出，支持多种常见格式"""
    h, w = orig_shape
    input_h, input_w = input_shape
    
    # 尝试确定输出格式
    if len(outputs) == 1:
        # 单一输出格式 (YOLO 风格)
        predictions = outputs[0]
        
        if len(predictions.shape) == 3:  # [batch, num_boxes, box_attrs]
            # 标准 YOLO 输出
            boxes = []
            scores = []
            class_ids = []
            
            for i in range(predictions.shape[1]):
                box_data = predictions[0, i, :]
                
                if box_data[4] < conf_threshold:  # 置信度过滤
                    continue
                
                # 获取类别和分数
                class_scores = box_data[5:]
                class_id = np.argmax(class_scores)
                score = class_scores[class_id] * box_data[4]  # 类别分数 * 置信度
                
                if score < conf_threshold:
                    continue
                
                # 边界框坐标 (中心点, 宽高)
                cx, cy, w_box, h_box = box_data[0:4]
                
                # 转换为左上角和右下角坐标
                x1 = (cx - w_box / 2) * w / input_w
                y1 = (cy - h_box / 2) * h / input_h
                x2 = (cx + w_box / 2) * w / input_w
                y2 = (cy + h_box / 2) * h / input_h
                
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                class_ids.append(class_id)
            
            if boxes:
                boxes = np.array(boxes)
                scores = np.array(scores)
                class_ids = np.array(class_ids)
                
                # 应用 NMS
                keep = nms(boxes, scores, iou_threshold)
                boxes = boxes[keep]
                scores = scores[keep]
                class_ids = class_ids[keep]
                
                return boxes, scores, class_ids
            else:
                return np.array([]), np.array([]), np.array([])
        
        elif len(predictions.shape) == 2:  # [num_boxes, box_attrs]
            # 另一种常见格式
            valid_detections = predictions[predictions[:, 4] >= conf_threshold]
            
            if len(valid_detections) == 0:
                return np.array([]), np.array([]), np.array([])
            
            boxes = []
            scores = []
            class_ids = []
            
            for detection in valid_detections:
                # 获取类别和分数
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                score = class_scores[class_id] * detection[4]
                
                if score < conf_threshold:
                    continue
                
                # 边界框坐标
                cx, cy, w_box, h_box = detection[0:4]
                
                # 转换为左上角和右下角坐标
                x1 = (cx - w_box / 2) * w / input_w
                y1 = (cy - h_box / 2) * h / input_h
                x2 = (cx + w_box / 2) * w / input_w
                y2 = (cy + h_box / 2) * h / input_h
                
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                class_ids.append(class_id)
            
            if boxes:
                boxes = np.array(boxes)
                scores = np.array(scores)
                class_ids = np.array(class_ids)
                
                # 应用 NMS
                keep = nms(boxes, scores, iou_threshold)
                boxes = boxes[keep]
                scores = scores[keep]
                class_ids = class_ids[keep]
                
                return boxes, scores, class_ids
            else:
                return np.array([]), np.array([]), np.array([])
    
    elif len(outputs) >= 3:
        # 分离的输出格式 (boxes, scores, classes)
        boxes = outputs[0][0]  # 假设第一个输出是边界框 [batch, num_boxes, 4]
        scores = outputs[1][0]  # 假设第二个输出是置信度 [batch, num_boxes]
        class_ids = outputs[2][0]  # 假设第三个输出是类别 [batch, num_boxes]
        
        # 如果有第四个输出，可能是有效检测数量
        num_detections = int(outputs[3][0]) if len(outputs) > 3 else len(boxes)
        
        # 截取有效检测
        boxes = boxes[:num_detections]
        scores = scores[:num_detections]
        class_ids = class_ids[:num_detections]
        
        # 过滤低置信度检测
        mask = scores >= conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        # 转换坐标到原始图像尺寸
        boxes[:, 0] *= w / input_w  # x1
        boxes[:, 1] *= h / input_h  # y1
        boxes[:, 2] *= w / input_w  # x2
        boxes[:, 3] *= h / input_h  # y2
        
        return boxes, scores, class_ids
    
    # 如果无法识别输出格式，返回空结果
    print(f"警告: 无法识别的输出格式，输出数量: {len(outputs)}")
    return np.array([]), np.array([]), np.array([])

def main(url, model_path):
    # 加载 ONNX 模型
    try:
        # 尝试使用 CUDA 提供程序
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        print(f"已加载 ONNX 模型 (CUDA): {model_path}")
    except Exception as e:
        print(f"CUDA 加载失败: {str(e)}")
        # 回退到 CPU
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        print(f"已加载 ONNX 模型 (CPU): {model_path}")
    
    # 获取模型输入输出信息
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"模型输入名称: {input_name}, 形状: {input_shape}")
    
    # 打开视频流
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"错误: 无法打开视频流 {url}")
        return
    
    # 获取原始帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"原始帧率: {fps} FPS, 尺寸: {w}x{h}")
    
    # 计算缩放参数
    target_size = 640  # 使用与 YOLO 相同的输入尺寸
    new_h, new_w, top, bottom, left, right = get_letterbox_params(h, w, target_size=target_size)
    if new_h is None:
        print(f"错误: get_letterbox_params 返回 None")
        return
    
    print(f"缩放后尺寸: {new_w}x{new_h}, padding: top={top}, bottom={bottom}, left={left}, right={right}")
    
    # FPS 计算相关变量
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 2.0  # 每2秒更新一次FPS
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧，退出...")
            break
        
        # 预处理
        processed = preprocess(frame, new_h, new_w, top, bottom, left, right)
        if processed is None:
            print("预处理失败，跳过当前帧")
            continue
        
        # 转换为模型输入格式
        try:
            # 检查 processed 的类型
            if isinstance(processed, torch.Tensor):
                # 如果是 PyTorch Tensor，转换为 numpy
                if processed.dim() == 4:
                    # 已经有批次维度
                    input_tensor = processed.float().div(255.0).cpu().numpy()
                else:
                    # 添加批次维度
                    input_tensor = processed.float().div(255.0).unsqueeze(0).cpu().numpy()
            else:
                # 如果是 numpy 数组
                input_tensor = processed.astype(np.float32) / 255.0  # 归一化
                
                # 确保输入是 NCHW 格式
                if input_tensor.ndim == 3:
                    # HWC to CHW
                    input_tensor = input_tensor.transpose(2, 0, 1)
                    # 添加批次维度
                    input_tensor = np.expand_dims(input_tensor, axis=0)
                elif input_tensor.ndim == 4 and input_tensor.shape[1] == 3:
                    # 已经是 NCHW 格式
                    pass
                else:
                    print(f"警告: 意外的输入形状: {input_tensor.shape}")
            
            # 确保内存连续
            input_tensor = np.ascontiguousarray(input_tensor)
            
            # 推理
            outputs = session.run(None, {input_name: input_tensor})
            
            # 处理输出
            boxes, scores, classes = process_outputs(
                outputs, 
                (input_tensor.shape[2], input_tensor.shape[3]),  # 输入高宽
                (h, w),  # 原始图像高宽
                conf_threshold=0.25
            )
            
            # 绘制检测结果
            annotated_frame = frame.copy()
            for box, score, cls in zip(boxes, scores, classes):
                if score < 0.5:  # 置信度阈值
                    continue
                
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(annotated_frame, 
                            (x1, y1), 
                            (x2, y2), 
                            (0, 255, 0), 2)
                cv2.putText(annotated_frame, 
                        f"Class {int(cls)}: {score:.2f}", 
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)
            
            # 显示帧率
            current_time = time.time()
            elapsed_total = current_time - start_time
            frame_count += 1
            
            if elapsed_total > 0:
                current_fps = frame_count / elapsed_total
                cv2.putText(annotated_frame, 
                        f"FPS: {current_fps:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow("ONNX Detection", annotated_frame)
            
            # 计算和显示FPS
            if elapsed_total >= fps_update_interval:
                processing_fps = frame_count / elapsed_total
                print(f"处理帧率: {processing_fps:.2f} FPS (frames: {frame_count})")
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
    print("处理完成")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python allinone.onnx.py <视频URL> <ONNX模型文件路径>")
        sys.exit(1)
    url = sys.argv[1]
    model_path = sys.argv[2]
    main(url, model_path) 