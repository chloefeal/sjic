import React, { useEffect, useRef, useState } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import io from 'socket.io-client';

function VideoPlayer({ streamUrl }) {
  const canvasRef = useRef(null);
  const socketRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // 创建WebSocket连接
    socketRef.current = io(process.env.REACT_APP_API_URL);
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // 监听视频帧数据
    socketRef.current.on('frame', (frameData) => {
      setLoading(false);
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      };
      img.src = `data:image/jpeg;base64,${frameData}`;
    });

    // 监听错误
    socketRef.current.on('stream_error', (error) => {
      setError(error.message);
      setLoading(false);
    });

    // 请求开始流式传输
    socketRef.current.emit('start_stream', { url: streamUrl });

    // 清理函数
    return () => {
      if (socketRef.current) {
        socketRef.current.emit('stop_stream');
        socketRef.current.disconnect();
      }
    };
  }, [streamUrl]);

  return (
    <Box sx={{ position: 'relative', width: '100%', aspectRatio: '16/9', bgcolor: 'black' }}>
      {loading && (
        <Box sx={{ 
          position: 'absolute', 
          top: '50%', 
          left: '50%', 
          transform: 'translate(-50%, -50%)'
        }}>
          <CircularProgress />
        </Box>
      )}
      {error && (
        <Box sx={{ 
          position: 'absolute', 
          top: '50%', 
          left: '50%', 
          transform: 'translate(-50%, -50%)',
          color: 'error.main'
        }}>
          <Typography>{error}</Typography>
        </Box>
      )}
      <canvas
        ref={canvasRef}
        style={{ 
          width: '100%', 
          height: '100%',
          display: loading || error ? 'none' : 'block'
        }}
        width={640}
        height={360}
      />
    </Box>
  );
}

export default VideoPlayer; 