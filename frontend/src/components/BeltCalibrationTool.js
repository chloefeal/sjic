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
  const containerRef = useRef(null);

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
  }
  // 停止视频流预览
  const stopStreaming = () => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
    // 如果有当前帧，保存为标定图像
    if (frameUrl) {
      setImageUrl(frameUrl);
      setFrameUrl(null);
    }
    setIsStreaming(false);
  };

  // 从视频流中截取当前帧
  const captureFrame = async () => {
    if (isStreaming && frameUrl) {
      // 直接使用当前的 frameUrl
      setImageUrl(frameUrl);
      setFrameUrl(null);
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

  // 处理画布点击
  const handleCanvasClick = (event) => {
    if (points.length >= 2) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    setPoints([...points, { x, y }]);
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
    points.forEach(point => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
      ctx.fillStyle = 'red';
      ctx.fill();
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
  const handleCalibrate = () => {
    if (points.length !== 2 || !beltWidth) return;

    // 计算像素距离
    const pixelWidth = Math.sqrt(
      Math.pow(points[1].x - points[0].x, 2) + 
      Math.pow(points[1].y - points[0].y, 2)
    );

    // 传递所有必要参数给后端
    onCalibrate({
      calibration: {
        belt_width: beltWidth,      // 真实宽度(cm)
        pixel_width: pixelWidth,    // 像素宽度
        frame_size: frameSize,      // 当前帧的尺寸
        points: points              // 标定点
      }
    });
    setOpen(false);
  };

  // 重置标定
  const handleReset = () => {
    setPoints([]);
    setBeltWidth(0);
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
    };
  }, []);

  return (
    <>
      <Button variant="outlined" onClick={() => setOpen(true)}>
        皮带标定
      </Button>

      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>皮带宽度标定</DialogTitle>
        <DialogContent>
          <Paper style={{ padding: 16, marginBottom: 16 }}>
            {!imageUrl && (  // 只在没有标定图像时显示这些按钮
              <>
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
                  disabled={!cameraId || !isStreaming}  // 只有在预览时才能截取
                  style={{ marginRight: 16 }}
                >
                  截取当前帧
                </Button>
              </>
            )}
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
              ref={containerRef}
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
              <Box 
                sx={{ 
                  width: '100%',
                  maxWidth: 800,
                  display: 'flex',
                  justifyContent: 'center'
                }}
              >
                <canvas
                  ref={canvasRef}
                  width={frameSize.width}
                  height={frameSize.height}
                  onClick={handleCanvasClick}
                  style={{ 
                    border: '1px solid #ccc', 
                    marginBottom: 16,
                    objectFit: 'contain'
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