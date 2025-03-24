#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet-18 模型训练和推理程序
支持对RTSP流进行实时检测
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import threading
import queue
import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ResNet18")

class CustomDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, data_dir, transform=None, is_train=True):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录，应包含train和val子目录，每个子目录下有分类目录
            transform: 图像变换
            is_train: 是否为训练集
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
        # 确定数据子目录
        self.data_subdir = os.path.join(data_dir, 'train' if is_train else 'val')
        
        # 获取所有类别
        self.classes = sorted([d for d in os.listdir(self.data_subdir) 
                              if os.path.isdir(os.path.join(self.data_subdir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 获取所有图像路径和标签
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.data_subdir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)
        
        logger.info(f"{'Training' if is_train else 'Validation'} dataset: {len(self.images)} images, {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 读取图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        num_epochs: 训练轮数
        device: 设备
    
    Returns:
        训练好的模型
    """
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        logger.info('-' * 10)
        
        # 每个epoch有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
                dataloader = train_loader
            else:
                model.eval()   # 评估模式
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # 迭代数据
            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # 如果是训练阶段，则反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 如果是验证阶段，且性能更好，则保存模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        
        logger.info('')
    
    time_elapsed = time.time() - since
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best val Acc: {best_acc:.4f}')
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, dataloader, class_names, device='cuda', num_images=6):
    """
    可视化模型预测结果
    
    Args:
        model: 模型
        dataloader: 数据加载器
        class_names: 类别名称
        device: 设备
        num_images: 可视化图像数量
    """
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}\ntrue: {class_names[labels[j]]}')
                
                # 将张量转换为图像
                inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                inp = std * inp + mean
                inp = np.clip(inp, 0, 1)
                
                ax.imshow(inp)
                
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.tight_layout()
                    plt.savefig('model_predictions.png')
                    return
        
        model.train(mode=was_training)
        plt.tight_layout()
        plt.savefig('model_predictions.png')

def save_model(model, save_path, class_names):
    """
    保存模型
    
    Args:
        model: 模型
        save_path: 保存路径
        class_names: 类别名称
    """
    # 创建目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, save_path)
    
    logger.info(f"Model saved to {save_path}")

def load_model(model_path, device='cuda'):
    """
    加载模型
    
    Args:
        model_path: 模型路径
        device: 设备
    
    Returns:
        模型和类别名称
    """
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    
    # 创建模型
    model = resnet18(weights=None)
    
    # 修改最后一层以适应类别数量
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 移动模型到设备
    model = model.to(device)
    
    # 设置为评估模式
    model.eval()
    
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Classes: {class_names}")
    
    return model, class_names

def inference(model, image_path, class_names, device='cuda'):
    """
    对单张图像进行推理
    
    Args:
        model: 模型
        image_path: 图像路径
        class_names: 类别名称
        device: 设备
    
    Returns:
        预测类别和置信度
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    
    # 预处理
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    
    return class_names[prediction.item()], confidence.item()

def inference_webcam(model, class_names, device='cuda', camera_id=0):
    """
    使用网络摄像头进行实时推理
    
    Args:
        model: 模型
        class_names: 类别名称
        device: 设备
        camera_id: 摄像头ID
    """
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        logger.error("无法打开摄像头")
        return
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        if not ret:
            logger.error("无法获取帧")
            break
        
        # 转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(rgb_frame)
        
        # 预处理
        image_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # 推理
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        # 显示结果
        pred_class = class_names[prediction.item()]
        conf = confidence.item()
        
        cv2.putText(frame, f"{pred_class}: {conf:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示帧
        cv2.imshow('ResNet-18 Inference', frame)
        
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

class RTSPVideoCapture:
    """RTSP视频捕获类，使用单独线程读取帧"""
    def __init__(self, rtsp_url, queue_size=128):
        self.rtsp_url = rtsp_url
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = time.time()
    
    def start(self):
        """启动RTSP流读取线程"""
        threading.Thread(target=self._update, daemon=True).start()
        return self
    
    def _update(self):
        """持续从RTSP流读取帧"""
        cap = cv2.VideoCapture(self.rtsp_url)
        
        if not cap.isOpened():
            logger.error(f"无法打开RTSP流: {self.rtsp_url}")
            self.stopped = True
            return
        
        logger.info(f"成功连接到RTSP流: {self.rtsp_url}")
        
        while not self.stopped:
            ret, frame = cap.read()
            
            if not ret:
                logger.error("无法获取帧，尝试重新连接...")
                cap.release()
                time.sleep(1)  # 等待1秒后重试
                cap = cv2.VideoCapture(self.rtsp_url)
                if not cap.isOpened():
                    logger.error(f"重新连接失败: {self.rtsp_url}")
                    self.stopped = True
                    break
                continue
            
            # 更新FPS计数
            self.frame_count += 1
            current_time = time.time()
            time_diff = current_time - self.last_fps_update
            
            if time_diff >= 1.0:  # 每秒更新一次FPS
                self.fps = self.frame_count / time_diff
                self.frame_count = 0
                self.last_fps_update = current_time
            
            # 如果队列已满，移除最旧的帧
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            
            # 添加新帧到队列
            self.queue.put(frame)
        
        cap.release()
    
    def read(self):
        """从队列中读取最新帧"""
        if self.stopped or self.queue.empty():
            return False, None
        
        return True, self.queue.get()
    
    def stop(self):
        """停止RTSP流读取"""
        self.stopped = True

def inference_rtsp(model, class_names, rtsp_url, output_path=None, device='cuda'):
    """
    对RTSP流进行实时推理
    
    Args:
        model: 模型
        class_names: 类别名称
        rtsp_url: RTSP URL
        output_path: 输出视频路径
        device: 设备
    """
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 创建RTSP视频捕获对象
    cap = RTSPVideoCapture(rtsp_url).start()
    
    # 等待连接建立
    time.sleep(2)
    
    # 视频写入器
    video_writer = None
    
    # 处理时间统计
    process_times = []
    
    try:
        while not cap.stopped:
            # 读取一帧
            ret, frame = cap.read()
            
            if not ret:
                logger.warning("无法获取帧，等待...")
                time.sleep(0.1)
                continue
            
            # 记录开始时间
            start_time = time.time()
            
            # 转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 转换为PIL图像
            pil_image = Image.fromarray(rgb_frame)
            
            # 预处理
            image_tensor = transform(pil_image).unsqueeze(0).to(device)
            
            # 推理
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probabilities, 1)
            
            # 计算处理时间
            process_time = time.time() - start_time
            process_times.append(process_time)
            
            # 显示结果
            pred_class = class_names[prediction.item()]
            conf = confidence.item()
            
            # 在帧上绘制信息
            cv2.putText(frame, f"Class: {pred_class}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {conf:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {cap.fps:.1f}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {process_time*1000:.1f}ms", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 添加时间戳
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (frame.shape[1] - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 初始化视频写入器
            if output_path is not None and video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    output_path, fourcc, 20.0,
                    (frame.shape[1], frame.shape[0])
                )
            
            # 写入帧
            if video_writer is not None:
                video_writer.write(frame)
            
            # 显示帧
            cv2.imshow('ResNet-18 RTSP Inference', frame)
            
            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        # 停止RTSP流读取
        cap.stop()
        
        # 释放视频写入器
        if video_writer is not None:
            video_writer.release()
        
        # 关闭窗口
        cv2.destroyAllWindows()
        
        # 打印统计信息
        if process_times:
            avg_time = sum(process_times) / len(process_times)
            logger.info(f"平均处理时间: {avg_time*1000:.2f}ms")
            logger.info(f"平均FPS: {1/avg_time:.2f}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ResNet-18 训练和推理')
    subparsers = parser.add_subparsers(dest='mode', help='操作模式')
    
    # 训练模式参数
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    train_parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    train_parser.add_argument('--num_epochs', type=int, default=25, help='训练轮数')
    train_parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    train_parser.add_argument('--pretrained', action='store_true', help='使用预训练模型')
    train_parser.add_argument('--output', type=str, default='resnet18_model.pth', help='输出模型路径')
    
    # 推理模式参数
    infer_parser = subparsers.add_parser('infer', help='使用模型进行推理')
    infer_parser.add_argument('--model', type=str, required=True, help='模型路径')
    infer_parser.add_argument('--image', type=str, help='图像路径')
    infer_parser.add_argument('--webcam', action='store_true', help='使用网络摄像头')
    infer_parser.add_argument('--camera_id', type=int, default=0, help='摄像头ID')
    infer_parser.add_argument('--rtsp', type=str, help='RTSP URL')
    infer_parser.add_argument('--output', type=str, help='输出视频路径')
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if args.mode == 'train':
        # 数据变换
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        # 创建数据集
        train_dataset = CustomDataset(args.data_dir, transform=data_transforms['train'], is_train=True)
        val_dataset = CustomDataset(args.data_dir, transform=data_transforms['val'], is_train=False)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # 获取类别名称
        class_names = train_dataset.classes
        
        # 创建模型
        if args.pretrained:
            logger.info("Using pre-trained model")
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            logger.info("Using model without pre-trained weights")
            model = resnet18(weights=None)
        
        # 修改最后一层以适应类别数量
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
        
        # 移动模型到设备
        model = model.to(device)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        
        # 学习率调度器
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # 训练模型
        model = train_model(model, train_loader, val_loader, criterion, optimizer, 
                           exp_lr_scheduler, num_epochs=args.num_epochs, device=device)
        
        # 可视化模型预测
        visualize_model(model, val_loader, class_names, device=device)
        
        # 保存模型
        save_model(model, args.output, class_names)
        
    elif args.mode == 'infer':
        # 加载模型
        model, class_names = load_model(args.model, device=device)
        
        if args.rtsp:
            # 使用RTSP流进行实时推理
            inference_rtsp(model, class_names, args.rtsp, args.output, device=device)
        elif args.webcam:
            # 使用网络摄像头进行实时推理
            inference_webcam(model, class_names, device=device, camera_id=args.camera_id)
        elif args.image:
            # 对单张图像进行推理
            pred_class, confidence = inference(model, args.image, class_names, device=device)
            logger.info(f"Prediction: {pred_class}, Confidence: {confidence:.4f}")
        else:
            logger.error("请指定图像路径、RTSP URL或使用网络摄像头")

if __name__ == "__main__":
    main() 