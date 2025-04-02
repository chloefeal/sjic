import sys
import os
import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
import argparse
from shapely.geometry import LineString, Polygon

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(description='皮带分割检测与跑偏分析')
    parser.add_argument('--source', type=str, required=True, help='视频源URL或文件路径')
    parser.add_argument('--model', type=str, required=True, help='分割模型路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--save', action='store_true', help='是否保存结果视频')
    parser.add_argument('--output', type=str, default='output.mp4', help='输出视频路径')
    parser.add_argument('--line1', type=str, help='边界线1坐标，格式: x1,y1,x2,y2')
    parser.add_argument('--line2', type=str, help='边界线2坐标，格式: x1,y1,x2,y2')
    parser.add_argument('--pause', action='store_true', help='找到皮带后暂停')
    return parser.parse_args()

def draw_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """在图像上绘制半透明掩码"""
    # 确保掩码与图像尺寸一致
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 创建彩色掩码
    colored_mask = np.zeros_like(image)
    mask_indices = mask == 1
    if np.any(mask_indices):  # 确保有掩码像素
        colored_mask[mask_indices] = color
    
    # 创建半透明叠加
    mask_image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    return mask_image

def parse_line(line_str):
    """解析命令行传入的线段坐标"""
    if not line_str:
        return None
    try:
        x1, y1, x2, y2 = map(int, line_str.split(','))
        return [(x1, y1), (x2, y2)]
    except:
        print(f"无法解析线段坐标: {line_str}，格式应为 x1,y1,x2,y2")
        return None

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scale_fill=False, scaleup=True):
    """
    将图像调整为指定大小，保持长宽比，并填充剩余部分
    
    参数:
        img: 输入图像
        new_shape: 目标尺寸 (高度, 宽度)
        color: 填充颜色
        auto: 是否自动计算最小比例
        scale_fill: 是否拉伸填充
        scaleup: 是否允许放大
    
    返回:
        调整大小后的图像，缩放比例，填充信息 (左, 上)
    """
    shape = img.shape[:2]  # 当前形状 [高度, 宽度]
    
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # 缩放比例 (新 / 旧)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 只缩小，不放大
        r = min(r, 1.0)
    
    # 计算填充
    ratio = r, r  # 宽度, 高度缩放比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh 填充
    
    if auto:  # 最小矩形
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh 填充
    elif scale_fill:  # 拉伸
        dw, dh = 0, 0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽度, 高度缩放比例
    
    dw /= 2  # 在两侧平均分配填充
    dh /= 2
    
    if shape[::-1] != new_unpad:  # 调整大小
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, ratio, (dw, dh)

def main():
    args = parse_args()
    
    # 加载模型
    try:
        model = YOLO(args.model)
        print(f"已加载模型: {args.model}")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return
    
    # 打开视频源
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"无法打开视频源: {args.source}")
        return
    
    print(f"视频源已打开: {args.source}")
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"视频尺寸: {width}x{height}, FPS: {fps}")
    
    # 解析边界线
    boundary_lines = []
    line1 = parse_line(args.line1)
    line2 = parse_line(args.line2)
    if line1:
        boundary_lines.append(line1)
    if line2:
        boundary_lines.append(line2)
    
    # 设置视频写入器
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # 处理计时
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取视频帧，退出")
                break
            
            # 打印帧信息，确认帧正常
            if frame_count == 0:
                print(f"帧尺寸: {frame.shape}, 类型: {frame.dtype}")
            
            # 创建可视化图像
            vis_frame = frame.copy()
            
            # 调整图像大小，保持长宽比
            img_resized, ratio, pad = letterbox(frame, new_shape=(640, 640))
            
            # 记录原始尺寸和调整后的尺寸
            if frame_count == 0:
                print(f"原始尺寸: {frame.shape}, 调整后尺寸: {img_resized.shape}")
                print(f"缩放比例: {ratio}, 填充: {pad}")
            
            # 使用调整大小后的图像进行推理
            results = model(img_resized, conf=args.conf)
            
            # 绘制边界线
            for line in boundary_lines:
                cv2.line(vis_frame, line[0], line[1], (0, 0, 255), 2)
            
            # 检查是否有分割结果
            is_deviation = False
            found_belt = False
            
            if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
                masks = results[0].masks
                if len(masks) > 0:
                    # 获取第一个掩码并转换为numpy数组
                    belt_mask = masks[0].data.cpu().numpy()[0]
                    
                    # 将掩码转换为二值图像
                    belt_mask_binary = (belt_mask > 0.5).astype(np.uint8)
                    
                    # 调整掩码尺寸以匹配原始图像
                    if belt_mask_binary.shape[:2] != frame.shape[:2]:
                        belt_mask_binary = cv2.resize(
                            belt_mask_binary, 
                            (frame.shape[1], frame.shape[0]), 
                            interpolation=cv2.INTER_NEAREST
                        )
                    
                    # 在图像上绘制掩码
                    vis_frame = draw_mask(vis_frame, belt_mask_binary)
                    
                    # 找到皮带的轮廓
                    contours, _ = cv2.findContours(
                        belt_mask_binary * 255, 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    if contours:
                        # 获取最大的轮廓
                        belt_contour = max(contours, key=cv2.contourArea)
                        
                        # 在图像上绘制轮廓
                        cv2.drawContours(vis_frame, [belt_contour], -1, (0, 255, 255), 2)
                        
                        found_belt = True
                        
                        # 检查皮带是否与边界线相交
                        if boundary_lines:
                            belt_polygon = Polygon(belt_contour.reshape(-1, 2))
                            for line in boundary_lines:
                                boundary_line = LineString(line)
                                if belt_polygon.intersects(boundary_line):
                                    is_deviation = True
                                    break
            
            # 显示跑偏状态
            status_text = "状态: 正常" if not is_deviation else "状态: 跑偏告警!"
            status_color = (0, 255, 0) if not is_deviation else (0, 0, 255)
            cv2.putText(
                vis_frame, 
                status_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                status_color, 
                2
            )
            
            # 计算并显示FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                current_fps = frame_count / elapsed
                fps_text = f"FPS: {current_fps:.1f}"
                start_time = time.time()
                frame_count = 0
            else:
                fps_text = f"FPS: {fps:.1f}"
            
            cv2.putText(
                vis_frame, 
                fps_text, 
                (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2
            )
            
            # 显示结果
            cv2.imshow("皮带分割与跑偏检测", vis_frame)
            
            # 保存视频
            if writer:
                writer.write(vis_frame)
            
            # 如果找到皮带并且设置了暂停标志，则暂停
            if found_belt and args.pause:
                print("找到皮带，暂停处理")
                cv2.waitKey(0)  # 等待任意键继续
            else:
                # 按q退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
