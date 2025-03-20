import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import cv2
import numpy as np
import time
import torch
import torchvision
from app.utils.calc import get_letterbox_params, preprocess

# COCO 类别名称
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    # ... 其他类别 ...
]

def main(url, model_name='mobile', input_size=384, skip_frames=2, conf_threshold=0.5):
    """
    使用优化的 Faster R-CNN 处理视频流
    
    参数:
        url: 视频流URL
        model_name: 模型类型 ('mobile', 'small', 'medium')
        input_size: 输入尺寸
        skip_frames: 跳帧数量
        conf_threshold: 置信度阈值
    """
    # 选择合适的模型
    print(f"正在加载 {model_name} Faster R-CNN 模型...")
    if model_name == 'mobile':
        # 使用 MobileNet 作为骨干网络 (最轻量)
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
            pretrained=True, 
            box_score_thresh=conf_threshold
        )
    elif model_name == 'small':
        # 使用 ResNet-18 作为骨干网络 (较轻量)
        backbone = torchvision.models.resnet18(pretrained=True)
        backbone.out_channels = 512
        
        anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        model = torchvision.models.detection.FasterRCNN(
            backbone,
            num_classes=91,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            box_score_thresh=conf_threshold
        )
    else:
        # 默认使用 ResNet-50 FPN (标准版)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
            box_score_thresh=conf_threshold
        )
    
    # 设置为评估模式
    model.eval()
    
    # 使用 GPU 如果可用，否则使用 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 使用混合精度
    use_fp16 = device.type == 'cuda'
    
    # 将模型移动到设备
    model = model.to(device)
    
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
    new_h, new_w, top, bottom, left, right = get_letterbox_params(h, w, target_size=input_size)
    
    # FPS 计算
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 2.0
    frame_index = 0
    
    # 处理视频帧
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧，退出...")
            break
        
        # 跳帧处理
        frame_index += 1
        if frame_index % (skip_frames + 1) != 0:
            continue
        
        try:
            # 预处理
            # 调整图像大小
            resized = cv2.resize(frame, (new_w, new_h))
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            
            # 转换为 RGB 并归一化
            input_tensor = padded[:, :, ::-1].astype(np.float32) / 255.0  # BGR to RGB
            input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).unsqueeze(0)  # HWC to NCHW
            input_tensor = input_tensor.to(device)
            
            # 使用混合精度推理
            with torch.cuda.amp.autocast(enabled=use_fp16):
                with torch.no_grad():
                    outputs = model(input_tensor)
            
            # 处理输出
            boxes = outputs[0]['boxes'].cpu().numpy()
            scores = outputs[0]['scores'].cpu().numpy()
            labels = outputs[0]['labels'].cpu().numpy()
            
            # 计算原始图像上的坐标
            scale_x = w / input_size
            scale_y = h / input_size
            offset_x = left
            offset_y = top
            
            # 绘制检测结果
            for box, score, label in zip(boxes, scores, labels):
                if score < conf_threshold:
                    continue
                
                # 转换坐标到原始图像
                x1, y1, x2, y2 = box
                x1 = int((x1 - offset_x) * scale_x)
                y1 = int((y1 - offset_y) * scale_y)
                x2 = int((x2 - offset_x) * scale_x)
                y2 = int((y2 - offset_y) * scale_y)
                
                # 确保坐标在图像范围内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                class_name = COCO_CLASSES[int(label) - 1] if int(label) - 1 < len(COCO_CLASSES) else f"Class {int(label)}"
                
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
            cv2.imshow("Faster R-CNN Optimized", frame)
            
            # 更新 FPS
            if elapsed >= fps_update_interval:
                print(f"处理帧率: {frame_count/elapsed:.2f} FPS")
                start_time = current_time
                frame_count = 0
                
            # 清理 GPU 缓存
            if device.type == 'cuda' and frame_index % 10 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"处理帧时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("处理完成")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python allinone.fasterrcnn_optimized.py <视频URL> [模型类型] [输入尺寸] [跳帧数] [置信度阈值]")
        print("模型类型: mobile (默认), small, medium")
        print("输入尺寸: 默认 384")
        print("跳帧数: 默认 2 (处理每3帧中的1帧)")
        print("置信度阈值: 默认 0.5")
        sys.exit(1)
    
    url = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'mobile'
    input_size = int(sys.argv[3]) if len(sys.argv) > 3 else 384
    skip_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    conf_threshold = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
    
    main(url, model_name, input_size, skip_frames, conf_threshold) 