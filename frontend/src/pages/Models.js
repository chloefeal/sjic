import React, { useState, useEffect } from 'react';
import { 
  Button, Dialog, DialogTitle, DialogContent, DialogActions, TextField,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow, 
  Paper, IconButton, Box, LinearProgress, Typography, Alert
} from '@mui/material';
import { Delete, Upload } from '@mui/icons-material';
import axios from '../utils/axios';

function Models() {
  const [models, setModels] = useState([]);
  const [openUpload, setOpenUpload] = useState(false);
  const [modelFile, setModelFile] = useState(null);
  const [modelName, setModelName] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await axios.get('/api/models');
      setModels(response || []);
      console.log('Models:', response);
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const handleUpload = async () => {
    if (!modelFile) {
      setError('请选择文件');
      return;
    }
    
    setUploading(true);
    setUploadProgress(0);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', modelFile);
    formData.append('name', modelName);
    
    console.log('FormData entries:');
    for (let pair of formData.entries()) {
      console.log(pair[0] + ': ' + pair[1]);
    }
    
    try {
      const response = await axios.post('/api/models/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(percentCompleted);
          console.log(`Upload progress: ${percentCompleted}%`);
        },
        timeout: 300000,
      });
      
      console.log('Upload response:', response);
      setUploading(false);
      setOpenUpload(false);
      setModelFile(null);
      setModelName('');
      fetchModels();
    } catch (error) {
      console.error('Error uploading model:', error);
      setError(error.response?.data?.error || '上传失败，请重试');
      setUploading(false);
    }
  };

  const handleDelete = async (id) => {
    try {
      await axios.delete(`/api/models/${id}`);
      fetchModels();
    } catch (error) {
      console.error('Error deleting model:', error);
    }
  };

  return (
    <Box>
      <Box sx={{ mb: 2 }}>
        <Button
          variant="contained"
          startIcon={<Upload />}
          onClick={() => setOpenUpload(true)}
        >
          上传模型
        </Button>
      </Box>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell>名称</TableCell>
              <TableCell>路径</TableCell>
              <TableCell>描述</TableCell>
              <TableCell>创建时间</TableCell>
              <TableCell>操作</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {models.map((model) => (
              <TableRow key={model.id}>
                <TableCell>{model.id}</TableCell>
                <TableCell>{model.name}</TableCell>
                <TableCell>{model.path}</TableCell>
                <TableCell>{model.description}</TableCell>
                <TableCell>{new Date(model.created_at).toLocaleString()}</TableCell>
                <TableCell>
                  <IconButton
                    color="error"
                    onClick={() => handleDelete(model.id)}
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

      <Dialog open={openUpload} onClose={() => setOpenUpload(false)}>
        <DialogTitle>上传模型</DialogTitle>
        <DialogContent>
          <TextField
            margin="dense"
            label="模型名称"
            fullWidth
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
          />
          <input
            type="file"
            accept=".pt,.pth,.weights,.engine,.onnx"
            onChange={(e) => setModelFile(e.target.files[0])}
            style={{ marginTop: '20px' }}
          />
          {uploading && (
            <div style={{ marginTop: '20px' }}>
              <Typography variant="body2" color="textSecondary">
                上传进度: {uploadProgress}%
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={uploadProgress} 
                sx={{ mt: 1 }}
              />
            </div>
          )}
          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenUpload(false)}>取消</Button>
          <Button 
            onClick={handleUpload} 
            disabled={!modelFile || uploading}
          >
            {uploading ? '上传中...' : '上传'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default Models; 