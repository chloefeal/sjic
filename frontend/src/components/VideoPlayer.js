import React, { useEffect, useRef, useState } from 'react';
import { Box } from '@mui/material';
import { io } from 'socket.io-client';

function VideoPlayer({ cameraId, onClose }) {
  const [frameUrl, setFrameUrl] = useState(null);
  const socketRef = useRef(null);
  const [frameSize, setFrameSize] = useState({ width: 800, height: 600 });

  // 计算缩放后的尺寸
  const calculateAspectRatio = (originalWidth, originalHeight, maxWidth = 800) => {
    const ratio = originalWidth / originalHeight;
    let width = maxWidth;
    let height = maxWidth / ratio;
    return { width, height };
  };

  useEffect(() => {
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

    socketRef.current.on('disconnect', (reason) => {
      console.log('Disconnected:', reason);
      if (socketRef.current) {
        socketRef.current.disconnect();
        socketRef.current = null;
      }
      if (frameUrl) {
        URL.revokeObjectURL(frameUrl);
      }
    });

    // 处理接收到的视频帧
    socketRef.current.on('frame', (frameData) => {
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
    });

    // 清理函数
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
      if (frameUrl) {
        URL.revokeObjectURL(frameUrl);
      }
    };
  }, [cameraId]);

  return (
    <Box
      sx={{
        width: '100%',
        maxWidth: frameSize.width,
        margin: '0 auto',
        position: 'relative'
      }}
    >
      {frameUrl && (
        <img
          src={frameUrl}
          alt="Camera Stream"
          style={{
            width: '100%',
            height: 'auto',
            display: 'block'
          }}
        />
      )}
    </Box>
  );
}

export default VideoPlayer; 