import React, { useState } from 'react';
import { Grid, Card, CardContent, Typography, Button, TextField, LinearProgress, Box } from '@mui/material';
import { CloudUpload, PlayArrow } from '@mui/icons-material';
import axios from '../utils/axios';

function Training() {
  const [dataset, setDataset] = useState(null);
  const [config, setConfig] = useState({
    epochs: 100,
    batchSize: 16,
    learningRate: 0.001,
  });
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleDatasetUpload = (event) => {
    const file = event.target.files[0];
    setDataset(file);
  };

  const startTraining = async () => {
    const formData = new FormData();
    formData.append('dataset', dataset);
    formData.append('config', JSON.stringify(config));

    try {
      setTraining(true);
      const response = await axios.post('/api/training/start', formData);
      
      // 模拟训练进度
      const interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 100) {
            clearInterval(interval);
            setTraining(false);
            return 100;
          }
          return prev + 1;
        });
      }, 1000);
    } catch (error) {
      console.error('Error starting training:', error);
      setTraining(false);
    }
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              数据集上传
            </Typography>
            <Button
              variant="contained"
              component="label"
              startIcon={<CloudUpload />}
            >
              上传数据集
              <input
                type="file"
                hidden
                accept=".zip"
                onChange={handleDatasetUpload}
              />
            </Button>
            {dataset && (
              <Typography variant="body2" sx={{ mt: 1 }}>
                已选择: {dataset.name}
              </Typography>
            )}
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              训练参数
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  label="训练轮数"
                  type="number"
                  value={config.epochs}
                  onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  label="批次大小"
                  type="number"
                  value={config.batchSize}
                  onChange={(e) => setConfig({ ...config, batchSize: parseInt(e.target.value) })}
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  label="学习率"
                  type="number"
                  value={config.learningRate}
                  onChange={(e) => setConfig({ ...config, learningRate: parseFloat(e.target.value) })}
                />
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Grid>
      {training && (
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                训练进度
              </Typography>
              <Box sx={{ width: '100%' }}>
                <LinearProgress variant="determinate" value={progress} />
                <Typography variant="body2" sx={{ mt: 1 }}>
                  {progress}%
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      )}
      <Grid item xs={12}>
        <Button
          variant="contained"
          startIcon={<PlayArrow />}
          onClick={startTraining}
          disabled={!dataset || training}
        >
          开始训练
        </Button>
      </Grid>
    </Grid>
  );
}

export default Training; 