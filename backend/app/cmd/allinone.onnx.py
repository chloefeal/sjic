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

def nms(boxes, scores, iou_threshold=0.45):
    """非极大值抑制的简单实现"""
    # 按照分数降序排序
    sorted_indices = np.argsort(scores)[::-1]
    
    keep = []
    while sorted_indices.size > 0:
        # 保留分数最高的框
        current = sorted_indices[0]
        keep.append(current)
        
        if sorted_indices.size == 1:
            break
            
        # 计算IoU
        current_box = boxes[current]
        other_boxes = boxes[sorted_indices[1:]]
        
        ious = compute_iou(current_box, other_boxes)
        
        # 保留IoU小于阈值的框
        mask = ious < iou_threshold
        sorted_indices = sorted_indices[1:][mask]
    
    return keep

def compute_iou(box, boxes):
    """计算一个框与多个框的IoU"""
    # 计算交集
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # 计算并集
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    
    return intersection / (union + 1e-6)

class OnnxModel:
    def __init__(self, model_path):
        # 设置 ONNX Runtime 选项
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 优先使用 CUDA 执行提供程序
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        try:
            # 创建 ONNX Runtime 会话
            self.session = ort.InferenceSession(model_path, options, providers=providers)
        except Exception as e:
            print(f"警告: 使用 CUDA 加载模型失败 ({str(e)})，尝试使用 CPU...")
            # 如果 CUDA 失败，尝试仅使用 CPU
            self.session = ort.InferenceSession(model_path, options, providers=['CPUExecutionProvider'])
        
        # 获取模型输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"模型输入: {self.input_name}, 形状: {self.input_shape}")
        print(f"模型输出: {self.output_names}")

    def __call__(self, input_tensor):
        # 运行推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return outputs

    def process_outputs(self, outputs):
        """处理模型输出，返回标准格式的检测结果"""
        # 这里需要根据实际模型的输出格式进行调整
        try:
            # 如果是标准 YOLO 输出格式
            if len(outputs) == 1 and outputs[0].shape[-1] == 85:  # YOLO 格式 (1, num_boxes, 85)
                predictions = outputs[0]
                boxes = predictions[..., :4]
                scores = predictions[..., 4:5] * predictions[..., 5:]
                classes = np.argmax(predictions[..., 5:], axis=-1)
                scores = np.max(scores, axis=-1)
                
                # 应用 NMS
                keep = nms(boxes[0], scores[0])
                
                return boxes[0][keep], scores[0][keep], classes[0][keep]
            
            # 如果输出已经是分离的格式
            elif len(outputs) == 3:  # 分离的输出格式 (boxes, scores, classes)
                return outputs[0][0], outputs[1][0], outputs[2][0]
            
            else:
                print(f"警告: 未知的输出格式，输出数量: {len(outputs)}")
                return None, None, None
                
        except Exception as e:
            print(f"处理输出时出错: {str(e)}")
            return None, None, None

def main(url, model_path):
    # 加载 ONNX 模型
    model = OnnxModel(model_path)
    print(f"已加载 ONNX 模型: {model_path}")

    # 打开 RTSP 流
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
    fps_update_interval = 2.0

    # 创建窗口
    cv2.namedWindow("ONNX Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("原始视频", cv2.WINDOW_NORMAL)
    cv2.namedWindow("预处理后", cv2.WINDOW_NORMAL)

    print("开始处理视频流...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧，退出...")
            break

        # 显示原始帧
        cv2.imshow("原始视频", frame)

        # 预处理
        processed = preprocess(frame, new_h, new_w, top, bottom, left, right)
        if processed is None:
            print("预处理失败，跳过当前帧")
            continue

        # 显示预处理后的帧
        cv2.imshow("预处理后", processed)

        # 转换为模型输入格式
        input_tensor = processed.astype(np.float32) / 255.0  # 归一化
        input_tensor = input_tensor.transpose(2, 0, 1)  # HWC to CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # 添加 batch 维度
        input_tensor = np.ascontiguousarray(input_tensor)  # 确保内存连续

        # 推理
        outputs = model(input_tensor)
        
        # 处理输出结果
        boxes, scores, classes = model.process_outputs(outputs)
        if boxes is None:
            continue

        # 绘制检测结果
        annotated_frame = processed.copy()
        for box, score, cls in zip(boxes, scores, classes):
            if score < 0.5:  # 置信度阈值
                continue
            
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated_frame, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         (0, 255, 0), 2)
            cv2.putText(annotated_frame, 
                       f"Class {int(cls)}: {score:.2f}", 
                       (int(x1), int(y1) - 10),
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