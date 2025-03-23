import React, { useState, useEffect, useRef } from 'react';
import { 
  Grid, Card, CardContent, Typography, Button, Dialog, DialogTitle, 
  DialogContent, DialogActions, TextField, Alert, Table, TableBody, 
  TableCell, TableContainer, TableHead, TableRow, Paper, IconButton 
} from '@mui/material';
import { Add, Visibility, Delete, Close } from '@mui/icons-material';
import axios from '../utils/axios';
import VideoPlayer from '../components/VideoPlayer';
import JSMpeg from '@cycjimmy/jsmpeg-player';

function VideoStreams() {
  const [streams, setStreams] = useState([]);
  const [openDialog, setOpenDialog] = useState(false);
  const [openPreview, setOpenPreview] = useState(false);
  const [selectedStream, setSelectedStream] = useState(null);
  const [newStream, setNewStream] = useState({ name: '', url: '' });
  const [error, setError] = useState(null);
  const [previewingCamera, setPreviewingCamera] = useState(null);

  useEffect(() => {
    fetchStreams();
  }, []);

  const fetchStreams = async () => {
    try {
      const response = await axios.get('/api/cameras');
      setStreams(response || []);
      console.log('Streams:', response);
    } catch (error) {
      console.error('Error fetching streams:', error);
      setError('Failed to load video streams');
      setStreams([]);
    }
  };

  const handleAddStream = async () => {
    try {
      const response = await axios.post('/api/cameras', newStream);
      setOpenDialog(false);
      setNewStream({ name: '', url: '' });
      fetchStreams();
    } catch (error) {
      console.error('Error adding stream:', error);
    }
  };

  const handlePreview = (camera) => {
    setPreviewingCamera(camera);
  };

  const handleClosePreview = () => {
    setPreviewingCamera(null);
  };

  const handleDelete = async (id) => {
    try {
      await axios.delete(`/api/cameras/${id}`);
      fetchStreams();
    } catch (error) {
      console.error('Error deleting stream:', error);
    }
  };

  return (
    <>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => setOpenDialog(true)}
            sx={{ mb: 2 }}
          >
            添加视频源
          </Button>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>ID</TableCell>
                  <TableCell>名称</TableCell>
                  <TableCell>URL</TableCell>
                  <TableCell>状态</TableCell>
                  <TableCell>创建时间</TableCell>
                  <TableCell>操作</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {streams.map((stream) => (
                  <TableRow key={stream.id}>
                    <TableCell>{stream.id}</TableCell>
                    <TableCell>{stream.name}</TableCell>
                    <TableCell>{stream.url}</TableCell>
                    <TableCell>{stream.status ? '在线' : '离线'}</TableCell>
                    <TableCell>{new Date(stream.created_at).toLocaleString()}</TableCell>
                    <TableCell>
                      <IconButton 
                        color="primary" 
                        onClick={() => handlePreview(stream)}
                        title="预览"
                      >
                        <Visibility />
                      </IconButton>
                      <IconButton 
                        color="error" 
                        onClick={() => handleDelete(stream.id)}
                        title="删除"
                      >
                        <Delete />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Grid>
      </Grid>

      {/* 添加视频源对话框 */}
      <Dialog open={openDialog} onClose={() => setOpenDialog(false)}>
        <DialogTitle>添加视频源</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="名称"
            fullWidth
            value={newStream.name}
            onChange={(e) => setNewStream({ ...newStream, name: e.target.value })}
          />
          <TextField
            margin="dense"
            label="RTSP地址"
            fullWidth
            value={newStream.url}
            onChange={(e) => setNewStream({ ...newStream, url: e.target.value })}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>取消</Button>
          <Button onClick={handleAddStream} color="primary">添加</Button>
        </DialogActions>
      </Dialog>

      {/* 预览对话框 */}
      <Dialog
        open={Boolean(previewingCamera)}
        onClose={handleClosePreview}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          预览 - {previewingCamera?.name}
          <IconButton
            onClick={handleClosePreview}
            sx={{ position: 'absolute', right: 8, top: 8 }}
          >
            <Close />
          </IconButton>
        </DialogTitle>
        <DialogContent>
          {previewingCamera && (
            <JSMpegPlayer cameraId={previewingCamera.id} />
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}

// 使用 JSMpeg 播放视频流
function JSMpegPlayer({ cameraId }) {
  const canvasRef = useRef(null);
  const playerRef = useRef(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    let player = null;
    
    const initPlayer = () => {
      if (!canvasRef.current) return;
      
      try {
        setLoading(true);
        // 获取 token
        const token = localStorage.getItem('token');
        
        // 构建 WebSocket URL
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = process.env.REACT_APP_API_URL 
          ? new URL(process.env.REACT_APP_API_URL).host 
          : window.location.host;
        
        const wsUrl = `${protocol}//${host}/ws/stream/${cameraId}?token=${token}`;
        console.log('Connecting to WebSocket URL:', wsUrl);
        
        // 创建 JSMpeg 播放器
        player = new JSMpeg.Player(wsUrl, {
          canvas: canvasRef.current,
          autoplay: true,
          audio: false,
          loop: true,
          protocols: ['binary'],  // 使用二进制 WebSocket
          videoBufferSize: 512 * 1024,
          onPlay: () => {
            console.log('Stream started playing');
            setLoading(false);
          },
          onError: (err) => {
            console.error('JSMpeg error:', err);
            setError(`播放错误: ${err}`);
            setLoading(false);
          }
        });
        
        playerRef.current = player;
        console.log('JSMpeg player initialized');
        
        // 添加超时检查
        setTimeout(() => {
          if (loading && !error) {
            setError('视频流加载超时，请刷新重试');
          }
        }, 10000);
      } catch (err) {
        console.error('Error initializing JSMpeg player:', err);
        setError(`初始化错误: ${err.message}`);
        setLoading(false);
      }
    };
    
    // 初始化播放器
    initPlayer();
    
    // 清理函数
    return () => {
      console.log('Cleaning up JSMpeg player');
      if (playerRef.current) {
        try {
          const p = playerRef.current;
          if (p && typeof p.destroy === 'function') {
            p.destroy();
          }
          playerRef.current = null;
        } catch (err) {
          console.error('Error destroying JSMpeg player:', err);
        }
      }
    };
  }, [cameraId]);
  
  return (
    <div className="video-container" style={{ textAlign: 'center' }}>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
          <Button 
            color="inherit" 
            size="small" 
            onClick={() => window.location.reload()}
            sx={{ ml: 2 }}
          >
            重试
          </Button>
        </Alert>
      )}
      {loading && !error && (
        <div style={{ marginBottom: '10px' }}>
          <Typography variant="body2" color="textSecondary">
            正在加载视频流...
          </Typography>
        </div>
      )}
      <canvas 
        ref={canvasRef} 
        width="640"
        height="360"
        style={{ 
          width: '100%', 
          maxWidth: '800px', 
          backgroundColor: '#000' 
        }}
      />
    </div>
  );
}

function HLSPlayer({ cameraId }) {
  const videoRef = useRef(null);
  
  useEffect(() => {
    if (videoRef.current) {
      const baseUrl = process.env.REACT_APP_API_URL || '';
      const hlsUrl = `${baseUrl}/api/hls/${cameraId}/playlist.m3u8`;
      
      if (videoRef.current.canPlayType('application/vnd.apple.mpegurl')) {
        // 原生 HLS 支持 (Safari)
        videoRef.current.src = hlsUrl;
      } else if (window.Hls) {
        // 使用 hls.js (Chrome, Firefox 等)
        const hls = new window.Hls();
        hls.loadSource(hlsUrl);
        hls.attachMedia(videoRef.current);
      }
    }
  }, [cameraId]);
  
  return (
    <div className="video-container" style={{ textAlign: 'center' }}>
      <video 
        ref={videoRef} 
        controls 
        autoPlay 
        style={{ 
          width: '100%', 
          maxWidth: '800px', 
          backgroundColor: '#000' 
        }}
      />
    </div>
  );
}

export default VideoStreams; 