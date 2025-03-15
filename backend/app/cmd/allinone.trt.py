import sys
import os
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import cv2
import numpy as np
import time
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from app.utils.calc import get_letterbox_params, preprocess

class TrtModel:
    def __init__(self, engine_path):
        # 加载 TRT 引擎
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        try:
            # 尝试加载 TensorRT 插件
            trt.init_libnvinfer_plugins(self.logger, '')
            print("TensorRT 插件已初始化")
            
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
            
            # 检测 TensorRT 版本并使用正确的 API
            print(f"TensorRT 版本: {trt.__version__}")
            
            # 获取所有张量名称
            tensor_names = []
            for i in range(self.engine.num_io_tensors):
                tensor_names.append(self.engine.get_tensor_name(i))
            
            print(f"张量名称: {tensor_names}")
            
            # 处理每个张量
            for i, name in enumerate(tensor_names):
                # 获取形状
                shape = self.engine.get_tensor_shape(name)
                print(f"张量 {name}: 形状 = {shape}")
                
                # 计算大小
                size = trt.volume(shape)
                
                # 获取数据类型
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                
                # 分配内存
                host_mem = cuda.pagelocked_empty(size, dtype)
                cuda_mem = cuda.mem_alloc(host_mem.nbytes)
                
                self.bindings.append(int(cuda_mem))
                
                # 确定是输入还是输出
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                
                if is_input:
                    print(f"张量 {name}: 输入")
                    self.host_inputs.append(host_mem)
                    self.cuda_inputs.append(cuda_mem)
                    self.input_shape = shape
                    self.input_name = name
                else:
                    print(f"张量 {name}: 输出")
                    self.host_outputs.append(host_mem)
                    self.cuda_outputs.append(cuda_mem)
                    self.output_shape = shape
                    self.output_name = name
                    
            print(f"TensorRT 引擎加载成功，输入数量: {len(self.host_inputs)}，输出数量: {len(self.host_outputs)}")
            
        except Exception as e:
            print(f"TensorRT 引擎加载失败: {str(e)}")
            raise

    def __call__(self, batch):
        try:
            # 将输入数据复制到页锁定内存
            np.copyto(self.host_inputs[0], batch.ravel())
            
            # 将输入数据从 CPU 传输到 GPU
            cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
            
            # 设置输入形状（如果需要）
            if hasattr(self, 'input_name'):
                self.context.set_input_shape(self.input_name, self.input_shape)
            
            # 执行推理
            self.context.execute_async_v2(self.bindings, self.stream.handle)
            
            # 将输出数据从 GPU 传输回 CPU
            for i in range(len(self.cuda_outputs)):
                cuda.memcpy_dtoh_async(self.host_outputs[i], self.cuda_outputs[i], self.stream)
                
            # 同步流
            self.stream.synchronize()
            
            # 重塑输出
            outputs = []
            for i, out in enumerate(self.host_outputs):
                # 尝试使用存储的形状
                if hasattr(self, 'output_shape'):
                    outputs.append(out.reshape(self.output_shape))
                else:
                    # 如果没有存储形状，尝试从引擎获取
                    outputs.append(out)  # 返回原始形状
            
            return outputs
            
        except Exception as e:
            print(f"推理过程中出错: {str(e)}")
            raise

def main(url, engine_path):
    # 加载 TRT 模型
    try:
        model = TrtModel(engine_path)
        print(f"已加载 TRT 模型: {engine_path}")
    except Exception as e:
        print(f"加载 TRT 模型失败: {str(e)}")
        print("尝试使用 ONNX 模型作为替代...")
        # 这里可以添加回退到 ONNX 模型的代码
        return

    # 打开 RTSP 流
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
    new_h, new_w, top, bottom, left, right = get_letterbox_params(h, w, target_size=640)
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

        # 打印预处理输出信息
        print(f"预处理输出形状: {processed.shape if hasattr(processed, 'shape') else type(processed)}")
        print(f"数据类型: {processed.dtype if hasattr(processed, 'dtype') else type(processed)}")

        # 转换为模型输入格式
        try:
            # 检查 processed 的类型
            if isinstance(processed, torch.Tensor):
                # 如果是 PyTorch Tensor，转换为 numpy
                # 移除批次维度，如果存在
                if processed.dim() == 4:
                    processed = processed.squeeze(0)  # 移除批次维度
                    print(f"移除批次维度后形状: {processed.shape}")
                    
                input_tensor = processed.float().div(255.0).cpu().numpy()
                
                # 确保输入是 CHW 格式
                if input_tensor.ndim == 3:
                    if input_tensor.shape[0] == 3:
                        # 已经是 CHW 格式
                        print("输入已经是 CHW 格式")
                    else:
                        # 需要从 HWC 转换为 CHW
                        input_tensor = input_tensor.transpose(2, 0, 1)
                        print(f"转换为 CHW 后形状: {input_tensor.shape}")
            else:
                # 如果是 numpy 数组
                input_tensor = processed.astype(np.float32) / 255.0  # 归一化
                
                # 确保输入是 CHW 格式
                if input_tensor.ndim == 3 and input_tensor.shape[2] == 3:
                    input_tensor = input_tensor.transpose(2, 0, 1)  # HWC to CHW
                    print(f"转换为 CHW 后形状: {input_tensor.shape}")
            
            # 添加批次维度
            if input_tensor.ndim == 3:
                input_tensor = np.expand_dims(input_tensor, axis=0)
                print(f"添加批次维度后形状: {input_tensor.shape}")
            
            # 确保内存连续
            input_tensor = np.ascontiguousarray(input_tensor)
            print(f"最终输入形状: {input_tensor.shape}")
            
            # 推理
            outputs = model(input_tensor)
            
            # 打印输出信息以便调试
            print(f"输出数量: {len(outputs)}")
            for i, out in enumerate(outputs):
                print(f"输出 {i} 形状: {out.shape}")

            # 处理输出结果（需要根据具体模型的输出格式调整）
            # 这里假设输出格式与 YOLO 类似
            if len(outputs) >= 3:
                boxes = outputs[0]  # 假设第一个输出是边界框
                scores = outputs[1]  # 假设第二个输出是置信度
                classes = outputs[2]  # 假设第三个输出是类别
            elif len(outputs) == 1:
                # 尝试解析单一输出
                out = outputs[0]
                if len(out.shape) >= 2 and out.shape[1] >= 7:  # 常见的检测输出格式
                    # 格式可能是 [batch, num_dets, 7] 其中 7 = [batch_id, x1, y1, x2, y2, score, class]
                    valid_dets = out[out[:, 0] > -1]  # 过滤无效检测
                    if len(valid_dets) > 0:
                        boxes = valid_dets[:, 1:5]
                        scores = valid_dets[:, 5]
                        classes = valid_dets[:, 6]
                    else:
                        continue
                else:
                    print(f"警告: 无法解析输出格式，形状: {out.shape}")
                    continue
            else:
                print(f"警告: 意外的输出格式，输出数量: {len(outputs)}")
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
            cv2.imshow("TensorRT Detection", annotated_frame)

            # 计算和显示FPS
            if elapsed_total >= fps_update_interval:
                processing_fps = frame_count / elapsed_total
                print(f"处理帧率: {processing_fps:.2f} FPS (frames: {frame_count})")
                start_time = current_time
                frame_count = 0
        except Exception as e:
            print(f"推理过程中出错: {str(e)}")
            continue

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("处理完成")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python allinone.trt.py <视频URL> <TRT引擎文件路径>")
        sys.exit(1)
    url = sys.argv[1]
    engine_path = sys.argv[2]
    main(url, engine_path) 