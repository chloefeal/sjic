import React, { useState, useRef, useEffect } from 'react';
import { 
  Button, Dialog, DialogTitle, DialogContent, DialogActions,
  TextField, Paper, Box
} from '@mui/material';
import axios from '../utils/axios';
import { io } from 'socket.io-client';  // 需要安装 socket.io-client

function BeltCalibrationTool({ cameraId, onCalibrate }) {
  const [open, setOpen] = useState(false);
  const [imageUrl, setImageUrl] = useState(null);
  const [points, setPoints] = useState([]);
  const [beltWidth, setBeltWidth] = useState(0); // 真实宽度(cm)
  const [isStreaming, setIsStreaming] = useState(false);
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const socketRef = useRef(null);
  const [frameUrl, setFrameUrl] = useState(null);  // 新增状态
  const [frameSize, setFrameSize] = useState({ width: 800, height: 600 });
  const lastFrameData = useRef(null);
  const [draggingPointIndex, setDraggingPointIndex] = useState(null);  // 新增：当前拖动的点索引

  // 计算缩放后的尺寸
  const calculateAspectRatio = (originalWidth, originalHeight, maxWidth = 800) => {
    const ratio = originalWidth / originalHeight;
    let width = maxWidth;
    let height = maxWidth / ratio;
    
    return { width, height };
  };

  // 开始视频流预览
  const startStreaming = () => {
    if (!cameraId) return;
    
    // 清除之前的标定图像和点
    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
      setImageUrl(null);
      setPoints([]);
      setBeltWidth(0);
    }
    
    setIsStreaming(true);
    
    // 创建 Socket.IO 连接
    socketRef.current = io(process.env.REACT_APP_API_URL, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: 5
    });

    // 连接成功后开始请求视频流
    socketRef.current.on('connect', () => {
      console.log('Connected to stream server');
      socketRef.current.emit('start_stream', { camera_id: cameraId });
    });

    socketRef.current.on('connect_error', (error) => {
      console.error('Connection error:', error);
      stopStreaming();
    });

    socketRef.current.on('disconnect', (reason) => {
      console.log('Disconnected:', reason);
      stopStreaming();
    });

    // 处理接收到的视频帧
    socketRef.current.on('frame', (frameData) => {
      drawFrame(frameData);
    });
  };
  
  const drawFrame = (frameData) => {
    // 保存原始的 frameData
    const blob = new Blob([frameData], { type: 'image/jpeg' });
    if (frameUrl) {
      URL.revokeObjectURL(frameUrl);
    }
    const url = URL.createObjectURL(blob);
    
    // 获取图像实际尺寸
    const img = new Image();
    img.onload = () => {
      const size = calculateAspectRatio(img.width, img.height);
      setFrameSize(size);
      URL.revokeObjectURL(img.src);
    };
    img.src = url;
    
    setFrameUrl(url);
    // 保存最新的 frameData 以便后续使用
    lastFrameData.current = frameData;
  };

  // 停止视频流预览
  const stopStreaming = () => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
    // 如果有当前帧，保存为标定图像
    if (lastFrameData.current) {
      const blob = new Blob([lastFrameData.current], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      setImageUrl(url);
    }
    if (frameUrl) {
      URL.revokeObjectURL(frameUrl);
      setFrameUrl(null);
    }
    setIsStreaming(false);
  };

  // 从视频流中截取当前帧
  const captureFrame = async () => {
    if (isStreaming && lastFrameData.current) {
      stopStreaming();
    } else {
      // 如果没有视频流，直接从摄像头获取图片
      try {
        const response = await axios.post('/api/cameras/capture', {
          camera_id: cameraId
        }, {
          responseType: 'blob'  // 确保响应类型是 blob
        });

        // 确保我们有一个有效的 Blob 对象
        if (response instanceof Blob) {
          const url = URL.createObjectURL(response);
          setImageUrl(url);
        } else {
          console.error('Invalid response data, not blob type:', response);
          const blob = new Blob([response], { type: 'image/jpeg' });
          const url = URL.createObjectURL(blob);
          setImageUrl(url);
        }
      } catch (error) {
        console.error('Error capturing frame:', error);
      }
    }
  };

  // 处理鼠标按下
  const handleMouseDown = (event) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // 检查是否点击了已存在的点
    const clickedPointIndex = points.findIndex(point => {
      const distance = Math.sqrt(
        Math.pow(point.x - x, 2) + Math.pow(point.y - y, 2)
      );
      return distance < 10;  // 10像素的点击范围
    });

    if (clickedPointIndex !== -1) {
      // 如果点击了已存在的点，开始拖动
      setDraggingPointIndex(clickedPointIndex);
    }
  };

  // 处理画布点击
  const handleCanvasClick = (event) => {
    if (draggingPointIndex !== null) {
      // 如果正在拖动，则点击时确认新位置
      setDraggingPointIndex(null);
      return;
    }

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    if (points.length < 2) {
      // 如果点数小于2，添加新点
      setPoints([...points, { x, y }]);
    }
  };

  // 处理鼠标移动
  const handleMouseMove = (event) => {
    if (draggingPointIndex === null) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(event.clientX - rect.left, canvas.width));
    const y = Math.max(0, Math.min(event.clientY - rect.top, canvas.height));

    // 更新点的位置
    const newPoints = [...points];
    newPoints[draggingPointIndex] = { x, y };
    setPoints(newPoints);
  };

  // 处理鼠标松开
  const handleMouseUp = () => {
    setDraggingPointIndex(null);
  };

  // 绘制画布
  const drawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const image = imageRef.current;

    // 清空画布
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 绘制图片
    if (image) {
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    }

    // 绘制点和线
    points.forEach((point, index) => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);  // 增大点的大小
      ctx.fillStyle = index === draggingPointIndex ? 'yellow' : 'red';  // 拖动时改变颜色
      ctx.fill();
      ctx.strokeStyle = 'white';  // 添加白色边框
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    if (points.length === 2) {
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      ctx.lineTo(points[1].x, points[1].y);
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  };

  // 计算标定结果
  const handleCalibrate = async () => {
    if (!imageUrl || points.length !== 2 || !beltWidth) return;

    try {
      // 获取当前图像的 blob 数据
      const response = await fetch(imageUrl);
      const blob = await response.blob();
      
      // 将 blob 转换为 base64
      const reader = new FileReader();
      reader.readAsDataURL(blob);
      
      reader.onloadend = () => {
        const base64data = reader.result;
        
        // 调用父组件的回调
        onCalibrate({
          calibration: {
            points: points,
            belt_width: beltWidth,
            pixel_width: Math.sqrt(
              Math.pow(points[1].x - points[0].x, 2) + 
              Math.pow(points[1].y - points[0].y, 2)
            ),
            frame_size: frameSize,
            frame: base64data  // 使用 base64 编码的图像数据
          }
        });
      };

      setOpen(false);
    } catch (error) {
      console.error('Error in calibration:', error);
    }
  };

  // 重置标定
  const handleReset = () => {
    setPoints([]);
    setBeltWidth(0);
    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
      setImageUrl(null);
    }
  };

  useEffect(() => {
    if (imageUrl && canvasRef.current) {
      const image = new Image();
      image.src = imageUrl;
      image.onload = () => {
        imageRef.current = image;
        drawCanvas();
      };
    }
  }, [imageUrl]);

  useEffect(() => {
    if (canvasRef.current) {
      drawCanvas();
    }
  }, [points]);

  // 组件卸载时清理
  useEffect(() => {
    return () => {
      stopStreaming();
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
      if (frameUrl) {
        URL.revokeObjectURL(frameUrl);
      }
      lastFrameData.current = null;
    };
  }, []);

  return (
    <>
      <Button variant="outlined" onClick={() => setOpen(true)}>
        皮带宽度标定
      </Button>

      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>皮带宽度标定</DialogTitle>
        <DialogContent>
          {/* 操作说明 */}
          <Paper style={{ 
            padding: 16, 
            marginBottom: 16, 
            backgroundColor: '#2f2f2f',
            color: '#ffffff'
          }}>
            <div style={{ 
              marginBottom: 8, 
              fontSize: '0.9rem', 
              fontWeight: 500 
            }}>
              操作说明：
            </div>
            <ol style={{ 
              margin: 0, 
              paddingLeft: 20,
              fontSize: '0.85rem'
            }}>
              <li>点击"开始预览"查看视频源画面</li>
              <li>点击"截取当前帧"或"停止预览"保存当前画面</li>
              <li>在图像上标记皮带两边的点（可拖动调整位置）：
                <ul style={{ 
                  fontSize: '0.8rem',
                  color: '#e0e0e0'
                }}>
                  <li>单击添加标定点（需要标记2个点）</li>
                  <li>拖动已有的点可以微调位置</li>
                </ul>
              </li>
              <li>输入皮带实际宽度(cm)</li>
              <li>点击"确定"完成标定</li>
            </ol>
          </Paper>

          {/* 操作按钮 */}
          <Paper style={{ padding: 16, marginBottom: 16 }}>
            <Button 
              variant="contained" 
              onClick={isStreaming ? stopStreaming : startStreaming}
              disabled={!cameraId}
              style={{ marginRight: 16 }}
            >
              {isStreaming ? '停止预览' : '开始预览'}
            </Button>
            <Button 
              variant="contained" 
              onClick={captureFrame}
              disabled={!cameraId || !isStreaming}
              style={{ marginRight: 16 }}
            >
              截取当前帧
            </Button>
            <Button 
              variant="outlined" 
              onClick={handleReset}
              disabled={!imageUrl}
            >
              重置
            </Button>
          </Paper>

          {/* 视频预览区域 */}
          {isStreaming && !imageUrl && frameUrl && (  // 确保有 frameUrl 时才显示
            <Box 
              mb={2} 
              sx={{ 
                width: '100%',
                maxWidth: 800,
                display: 'flex',
                justifyContent: 'center'
              }}
            >
              <img
                src={frameUrl}
                width={frameSize.width}
                height={frameSize.height}
                style={{ 
                  border: '1px solid #ccc',
                  objectFit: 'contain'
                }}
                alt="Video stream"
              />
            </Box>
          )}

          {/* 标定区域 */}
          {imageUrl && (
            <>
              <Box sx={{ 
                width: '100%',
                maxWidth: 800,
                display: 'flex',
                justifyContent: 'center'
              }}>
                <canvas
                  ref={canvasRef}
                  width={frameSize.width}
                  height={frameSize.height}
                  onClick={handleCanvasClick}
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  onMouseLeave={handleMouseUp}
                  style={{ 
                    border: '1px solid #ccc', 
                    marginBottom: 16,
                    objectFit: 'contain',
                    cursor: draggingPointIndex !== null ? 'grabbing' : 'crosshair'
                  }}
                />
              </Box>
              <TextField
                fullWidth
                type="number"
                label="皮带实际宽度(cm)"
                value={beltWidth}
                onChange={(e) => setBeltWidth(parseFloat(e.target.value))}
                inputProps={{ min: 0, step: 0.1 }}
              />
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)}>取消</Button>
          <Button 
            onClick={handleCalibrate}
            variant="contained"
            disabled={points.length !== 2 || !beltWidth}
          >
            确定
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

export default BeltCalibrationTool; 