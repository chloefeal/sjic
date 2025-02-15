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
  const videoRef = useRef(null);
  const socketRef = useRef(null);

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
      reconnectionAttempts: 5,
      path: '/stream'
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
      const videoElement = videoRef.current;
      if (videoElement) {
        const blob = new Blob([frameData], { type: 'image/jpeg' });
        const url = URL.createObjectURL(blob);
        videoElement.src = url;
        // 清理旧的 URL
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }
    });
  };

  // 停止视频流预览
  const stopStreaming = () => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
    setIsStreaming(false);
  };

  // 从视频流中截取当前帧
  const captureFrame = async () => {
    if (isStreaming && videoRef.current) {
      // 从视频元素创建 canvas
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(videoRef.current, 0, 0);
      
      // 转换为 blob
      canvas.toBlob((blob) => {
        setImageUrl(URL.createObjectURL(blob));
      }, 'image/jpeg');
      
      // 停止视频流
      stopStreaming();
    } else {
      // 如果没有视频流，使用原来的方式获取图片
      try {
        const response = await axios.post('/api/cameras/capture', {
          camera_id: cameraId
        }, {
          responseType: 'blob'
        });
        setImageUrl(URL.createObjectURL(response.data));
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

    // 计算像素到厘米的转换比例
    const pixel_to_cm = beltWidth / pixelWidth;

    onCalibrate({
      pixel_to_cm,
      calibration: {
        belt_width: beltWidth,
        points: points
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
            {!imageUrl && (
              <Button 
                variant="contained" 
                onClick={isStreaming ? stopStreaming : startStreaming}
                disabled={!cameraId}
                style={{ marginRight: 16 }}
              >
                {isStreaming ? '停止预览' : '开始预览'}
              </Button>
            )}
            <Button 
              variant="contained" 
              onClick={captureFrame}
              disabled={!cameraId}
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
          {isStreaming && !imageUrl && (
            <Box mb={2}>
              <video
                ref={videoRef}
                width={800}
                height={600}
                autoPlay
                style={{ border: '1px solid #ccc' }}
              />
            </Box>
          )}

          {/* 标定区域 */}
          {imageUrl && (
            <>
              <canvas
                ref={canvasRef}
                width={800}
                height={600}
                onClick={handleCanvasClick}
                style={{ border: '1px solid #ccc', marginBottom: 16 }}
              />
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