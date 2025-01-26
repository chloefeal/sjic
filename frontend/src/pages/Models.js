import React, { useState, useEffect } from 'react';
import { Grid, Card, CardContent, Typography, Button, IconButton, Dialog, DialogTitle, DialogContent, DialogActions, TextField } from '@mui/material';
import { Delete, Upload } from '@mui/icons-material';
import axios from '../utils/axios';

function Models() {
  const [models, setModels] = useState([]);
  const [openUpload, setOpenUpload] = useState(false);
  const [modelFile, setModelFile] = useState(null);
  const [modelName, setModelName] = useState('');

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await axios.get('/api/models');
      setModels(response || []);  // 确保始终是数组
      console.log('Models:', response);
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', modelFile);
    formData.append('name', modelName);

    try {
      await axios.post('/api/models/upload', formData);
      setOpenUpload(false);
      setModelFile(null);
      setModelName('');
      fetchModels();
    } catch (error) {
      console.error('Error uploading model:', error);
    }
  };

  return (
    <>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Button
            variant="contained"
            startIcon={<Upload />}
            onClick={() => setOpenUpload(true)}
          >
            上传模型
          </Button>
        </Grid>
        {models.map((model) => (
          <Grid item xs={12} md={6} lg={4} key={model.id}>
            <Card>
              <CardContent>
                <Typography variant="h6">{model.name}</Typography>
                <Typography variant="body2" color="textSecondary">
                  {model.description}
                </Typography>
                <IconButton
                  color="error"
                  onClick={() => {/* 删除模型 */}}
                >
                  <Delete />
                </IconButton>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Dialog open={openUpload} onClose={() => setOpenUpload(false)}>
        <DialogTitle>上传模型</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="模型名称"
            fullWidth
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
          />
          <input
            type="file"
            accept=".pt,.pth"
            onChange={(e) => setModelFile(e.target.files[0])}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenUpload(false)}>取消</Button>
          <Button onClick={handleUpload} color="primary">
            上传
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

export default Models; 