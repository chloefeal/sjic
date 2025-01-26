import React, { useState, useEffect } from 'react';
import { 
  Grid, Card, CardContent, Typography, Button, Dialog, DialogTitle, 
  DialogContent, DialogActions, TextField, Alert, Table, TableBody, 
  TableCell, TableContainer, TableHead, TableRow, Paper, IconButton 
} from '@mui/material';
import { Add, Visibility, Delete } from '@mui/icons-material';
import axios from '../utils/axios';
import VideoPlayer from '../components/VideoPlayer';

function VideoStreams() {
  const [streams, setStreams] = useState([]);
  const [openDialog, setOpenDialog] = useState(false);
  const [openPreview, setOpenPreview] = useState(false);
  const [selectedStream, setSelectedStream] = useState(null);
  const [newStream, setNewStream] = useState({ name: '', url: '' });
  const [error, setError] = useState(null);

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

  const handlePreview = (stream) => {
    setSelectedStream(stream);
    setOpenPreview(true);
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
        open={openPreview} 
        onClose={() => setOpenPreview(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>{selectedStream?.name}</DialogTitle>
        <DialogContent>
          {selectedStream && <VideoPlayer streamUrl={selectedStream.url} />}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenPreview(false)}>关闭</Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

export default VideoStreams; 