import React, { useState, useEffect } from 'react';
import { Grid, Card, CardContent, Typography, Button, FormControl, InputLabel, Select, MenuItem, TextField, Box, Slider } from '@mui/material';
import { Save } from '@mui/icons-material';
import axios from '../utils/axios';

function AlgorithmSettings() {
  const [settings, setSettings] = useState({
    modelId: '',
    cameraId: '',
    confidence: 0.5,
    regions: [],
    alertThreshold: 3,
  });

  const [models, setModels] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [drawing, setDrawing] = useState(false);

  useEffect(() => {
    fetchModels();
    fetchCameras();
    fetchSettings();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await axios.get('/api/models');
      setModels(response || []);
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const fetchCameras = async () => {
    try {
      const response = await axios.get('/api/cameras');
      setCameras(response || []);
    } catch (error) {
      console.error('Error fetching cameras:', error);
    }
  };

  const fetchSettings = async () => {
    try {
      const response = await axios.get('/api/settings');
      if (response) {
        setSettings(prev => ({
          ...prev,
          ...response
        }));
      }
    } catch (error) {
      console.error('Error fetching settings:', error);
    }
  };

  const handleSave = async () => {
    try {
      await axios.put('/api/settings', settings);
    } catch (error) {
      console.error('Error saving settings:', error);
    }
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              检测设置
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>选择模型</InputLabel>
                  <Select
                    value={settings.modelId || ''}
                    onChange={(e) => setSettings({ ...settings, modelId: e.target.value })}
                  >
                    {models.map((model) => (
                      <MenuItem key={model.id} value={model.id}>
                        {model.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>选择摄像头</InputLabel>
                  <Select
                    value={settings.cameraId || ''}
                    onChange={(e) => setSettings({ ...settings, cameraId: e.target.value })}
                  >
                    {cameras.map((camera) => (
                      <MenuItem key={camera.id} value={camera.id}>
                        {camera.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="置信度阈值"
                  type="number"
                  value={settings.confidence}
                  onChange={(e) => setSettings({ ...settings, confidence: parseFloat(e.target.value) })}
                  inputProps={{ step: 0.1, min: 0, max: 1 }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="告警阈值"
                  type="number"
                  value={settings.alertThreshold}
                  onChange={(e) => setSettings({ ...settings, alertThreshold: parseInt(e.target.value) })}
                  inputProps={{ min: 1 }}
                />
              </Grid>
              <Grid item xs={12}>
                <Button variant="contained" color="primary" onClick={handleSave}>
                  保存设置
                </Button>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              区域设置
            </Typography>
            <Box
              sx={{
                width: '100%',
                height: '400px',
                border: '1px solid #ccc',
                position: 'relative',
              }}
            >
              {/* 这里添加区域绘制功能 */}
            </Box>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
}

export default AlgorithmSettings; 