import sys
import os
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
from app.utils.calc import get_letterbox_params, preprocess

def main(url, model_path):
    # 加载 YOLO 模型
    try:
        model = YOLO(model_path)
        print(f"已加载 YOLO 模型: {model_path}")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return
    
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
        
        try:
            # 使用 YOLO 直接处理帧
            results = model(frame, conf=0.25)
            
            # 绘制检测结果
            annotated_frame = results[0].plot()
            
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
            cv2.imshow("YOLO Detection", annotated_frame)
            
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
        print("用法: python allinone.yolo.py <视频URL> <模型文件路径>")
        sys.exit(1)
    url = sys.argv[1]
    model_path = sys.argv[2]
    main(url, model_path) 