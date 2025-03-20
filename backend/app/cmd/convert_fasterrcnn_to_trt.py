import torch
import torchvision
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class FasterRCNNWrapper(torch.nn.Module):
    """包装 Faster R-CNN 模型以便于导出到 ONNX"""
    def __init__(self, model):
        super(FasterRCNNWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        # 在推理模式下运行模型
        self.model.eval()
        with torch.no_grad():
            # 获取预测结果
            detections = self.model(x)[0]  # 只取第一个批次的结果
            
            # 提取边界框、分数和标签
            boxes = detections['boxes']
            scores = detections['scores']
            labels = detections['labels']
            
            # 确保输出形状一致
            max_detections = 100  # 最大检测数量
            
            # 如果检测数量少于最大值，用零填充
            if len(boxes) < max_detections:
                zeros_boxes = torch.zeros((max_detections - len(boxes), 4), device=boxes.device)
                zeros_scores = torch.zeros(max_detections - len(scores), device=scores.device)
                zeros_labels = torch.zeros(max_detections - len(labels), dtype=torch.int64, device=labels.device)
                
                boxes = torch.cat([boxes, zeros_boxes], dim=0)
                scores = torch.cat([scores, zeros_scores], dim=0)
                labels = torch.cat([labels, zeros_labels], dim=0)
            else:
                # 如果检测数量超过最大值，只保留前 max_detections 个
                boxes = boxes[:max_detections]
                scores = scores[:max_detections]
                labels = labels[:max_detections]
            
            return boxes, labels, scores

def convert_fasterrcnn_to_trt(output_path, precision='fp16'):
    """
    将 Faster R-CNN 模型转换为 TensorRT 引擎
    
    参数:
        output_path: 输出的 TensorRT 引擎文件路径
        precision: 精度模式，可选 'fp32', 'fp16', 'int8'
    """
    print(f"正在加载 Faster R-CNN 模型...")
    # 加载预训练的 Faster R-CNN 模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
    model.eval()
    
    # 包装模型以便于导出
    wrapped_model = FasterRCNNWrapper(model)
    wrapped_model.eval()
    
    # 将模型移动到 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wrapped_model = wrapped_model.to(device)
    
    # 创建一个示例输入
    dummy_input = torch.randn(1, 3, 640, 640, device=device)
    
    # 首先导出为 ONNX 格式
    onnx_path = output_path.replace('.engine', '.onnx')
    print(f"正在导出 ONNX 模型到 {onnx_path}...")
    
    # 导出为 ONNX 格式，使用固定输出形状
    torch.onnx.export(
        wrapped_model, 
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['boxes', 'labels', 'scores'],
        dynamic_axes={
            'input': {0: 'batch_size'},  # 只允许批次大小是动态的
        }
    )
    
    print(f"ONNX 模型导出完成，正在转换为 TensorRT...")
    
    # 使用 trtexec 工具将 ONNX 模型转换为 TensorRT 引擎
    precision_flag = ""
    if precision == 'fp16':
        precision_flag = "--fp16"
    elif precision == 'int8':
        precision_flag = "--int8"
    
    # 添加更多参数以处理动态形状
    os.system(f"trtexec --onnx={onnx_path} --saveEngine={output_path} \
              --workspace=4096 {precision_flag} --verbose \
              --minShapes=input:1x3x640x640 --optShapes=input:1x3x640x640 \
              --maxShapes=input:1x3x640x640")
    
    print(f"TensorRT 引擎已保存到 {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python convert_fasterrcnn_to_trt.py <输出引擎路径> [精度]")
        sys.exit(1)
    
    output_path = sys.argv[1]
    precision = sys.argv[2] if len(sys.argv) > 2 else 'fp16'
    
    convert_fasterrcnn_to_trt(output_path, precision) 