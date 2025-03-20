import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import cv2
import numpy as np
import time
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from app.utils.calc import get_letterbox_params, preprocess

# COCO 类别名称
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class TrtFasterRCNN:
    def __init__(self, engine_path):
        # 加载 TRT 引擎
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        try:
            # 初始化 TensorRT 插件
            trt.init_libnvinfer_plugins(self.logger, '')
            print("TensorRT 插件已初始化")
            
            # 加载引擎
            with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
                
            if not self.engine:
                raise RuntimeError("引擎加载失败")
                
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            
            # 分配内存
            self.host_inputs = []
            self.host_outputs = []
            self.cuda_inputs = []
            self.cuda_outputs = []
            self.bindings = []
            self.output_shapes = []
            
            # 处理输入输出
            for binding in range(self.engine.num_bindings):
                size = trt.volume(self.engine.get_binding_shape(binding))
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                
                # 分配内存
                host_mem = cuda.pagelocked_empty(size, dtype)
                cuda_mem = cuda.mem_alloc(host_mem.nbytes)
                
                self.bindings.append(int(cuda_mem))
                
                if self.engine.binding_is_input(binding):
                    self.host_inputs.append(host_mem)
                    self.cuda_inputs.append(cuda_mem)
                    self.input_shape = self.engine.get_binding_shape(binding)
                    self.input_idx = binding
                else:
                    self.host_outputs.append(host_mem)
                    self.cuda_outputs.append(cuda_mem)
                    self.output_shapes.append(self.engine.get_binding_shape(binding))
                    
            print(f"TensorRT 引擎加载成功，输入形状: {self.input_shape}, 输出数量: {len(self.host_outputs)}")
            
        except Exception as e:
            print(f"TensorRT 引擎加载失败: {str(e)}")
            raise
    
    def __call__(self, image):
        try:
            # 预处理图像
            input_tensor = self.preprocess(image)
            
            # 设置动态输入形状
            if -1 in self.input_shape:
                self.context.set_binding_shape(self.input_idx, input_tensor.shape)
            
            # 将输入数据复制到设备
            cuda.memcpy_htod_async(self.cuda_inputs[0], input_tensor.ravel(), self.stream)
            
            # 执行推理
            self.context.execute_async_v2(self.bindings, self.stream.handle, None)
            
            # 将输出数据复制回主机
            outputs = []
            for i, output in enumerate(self.cuda_outputs):
                cuda.memcpy_dtoh_async(self.host_outputs[i], output, self.stream)
                
            # 同步流
            self.stream.synchronize()
            
            # 后处理输出
            boxes, labels, scores = self.postprocess(self.host_outputs)
            
            return boxes, labels, scores
            
        except Exception as e:
            print(f"推理过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], [], []
    
    def preprocess(self, image):
        # 调整图像大小为模型输入尺寸
        h, w = image.shape[:2]
        input_h, input_w = 640, 640  # 假设模型输入尺寸为 640x640
        
        # 计算缩放比例
        scale = min(input_h / h, input_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h))
        
        # 创建画布并将图像放在中心
        canvas = np.zeros((input_h, input_w, 3), dtype=np.float32)
        canvas[(input_h - new_h) // 2:(input_h - new_h) // 2 + new_h, 
               (input_w - new_w) // 2:(input_w - new_w) // 2 + new_w, :] = resized
        
        # 转换为 RGB 并归一化
        canvas = canvas[:, :, ::-1]  # BGR to RGB
        canvas = canvas / 255.0  # 归一化到 [0, 1]
        
        # 转换为 NCHW 格式
        canvas = canvas.transpose(2, 0, 1)
        canvas = np.expand_dims(canvas, axis=0)
        
        # 确保内存连续
        return np.ascontiguousarray(canvas, dtype=np.float32)
    
    def postprocess(self, outputs):
        # 解析输出
        # 假设输出格式为: [boxes, labels, scores]
        boxes = outputs[0].reshape(-1, 4)
        labels = outputs[1].reshape(-1)
        scores = outputs[2].reshape(-1)
        
        # 过滤低置信度检测
        mask = scores > 0.5
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        
        return boxes, labels, scores

def main(url, engine_path):
    # 加载 TensorRT Faster R-CNN 模型
    try:
        model = TrtFasterRCNN(engine_path)
        print(f"已加载 TensorRT Faster R-CNN 模型: {engine_path}")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return
    
    # 打开视频流
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"错误: 无法打开视频流 {url}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"原始帧率: {fps} FPS, 尺寸: {w}x{h}")
    
    # FPS 计算
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 2.0
    
    # 处理视频帧
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧，退出...")
            break
        
        try:
            # 推理
            boxes, labels, scores = model(frame)
            
            # 绘制检测结果
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box.astype(int)
                class_name = COCO_CLASSES[int(label)] if int(label) < len(COCO_CLASSES) else f"Class {int(label)}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name}: {score:.2f}", 
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 显示帧率
            current_time = time.time()
            elapsed = current_time - start_time
            frame_count += 1
            
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow("Faster R-CNN TensorRT", frame)
            
            # 更新 FPS
            if elapsed >= fps_update_interval:
                print(f"处理帧率: {frame_count/elapsed:.2f} FPS")
                start_time = current_time
                frame_count = 0
                
        except Exception as e:
            print(f"处理帧时出错: {str(e)}")
            continue
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("处理完成")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python allinone.fasterrcnn_trt.py <视频URL> <TensorRT引擎文件路径>")
        sys.exit(1)
    url = sys.argv[1]
    engine_path = sys.argv[2]
    main(url, engine_path) 