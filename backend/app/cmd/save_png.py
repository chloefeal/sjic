import cv2
import os
import time
import datetime
import threading
import sys
from pathlib import Path

def save_frame(rtsp_url, save_dir, interval=60):
    """
    从RTSP流中每隔指定时间保存一帧图像
    
    参数:
        rtsp_url: RTSP视频流地址
        save_dir: 保存图像的目录
        interval: 保存间隔，单位为秒，默认60秒
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"开始处理视频流: {rtsp_url} -> 保存到: {save_dir}")
    
    # 连接失败计数
    fail_count = 0
    max_fails = 5
    
    while True:
        try:
            # 打开视频流
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                print(f"无法打开视频流: {rtsp_url}")
                fail_count += 1
                if fail_count >= max_fails:
                    print(f"连接失败次数过多，停止处理: {rtsp_url}")
                    break
                time.sleep(10)  # 等待10秒后重试
                continue
            
            # 重置失败计数
            fail_count = 0
            
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                print(f"无法读取视频帧: {rtsp_url}")
                cap.release()
                time.sleep(5)  # 等待5秒后重试
                continue
            
            # 生成文件名 (yymmddhhmmss.png)
            timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
            filename = f"{timestamp}.png"
            filepath = os.path.join(save_dir, filename)
            
            # 保存图像
            cv2.imwrite(filepath, frame)
            print(f"已保存图像: {filepath}")
            
            # 释放资源
            cap.release()
            
            # 等待指定的间隔时间
            time.sleep(interval)
            
        except Exception as e:
            print(f"处理视频流时出错 {rtsp_url}: {str(e)}")
            time.sleep(5)  # 出错后等待5秒再重试

def main(rtsp_file):
    """
    主函数，读取RTSP文件并为每个流创建线程
    
    参数:
        rtsp_file: 包含RTSP URL和保存目录的文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(rtsp_file):
        print(f"文件不存在: {rtsp_file}")
        return
    
    # 读取文件内容
    with open(rtsp_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 创建线程列表
    threads = []
    
    # 为每个RTSP流创建线程
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # 分割RTSP URL和保存目录
        parts = line.split(' ', 1)
        if len(parts) < 2:
            print(f"格式错误，跳过: {line}")
            continue
        
        rtsp_url = parts[0]
        save_dir = parts[1]
        
        # 创建并启动线程
        thread = threading.Thread(
            target=save_frame, 
            args=(rtsp_url, save_dir),
            daemon=True
        )
        thread.start()
        threads.append(thread)
    
    # 等待所有线程完成（实际上不会完成，因为线程会一直运行）
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("程序被用户中断，正在退出...")
        sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        rtsp_file = sys.argv[1]
    else:
        # 默认使用当前目录下的rtsp.txt
        rtsp_file = "rtsp.txt"
    
    main(rtsp_file) 