import React, { useState, useEffect } from 'react';
import { 
  Grid, Card, CardContent, Typography, Button, Dialog, DialogTitle, 
  DialogContent, DialogActions, TextField, FormControl, InputLabel, 
  Select, MenuItem, Table, TableBody, TableCell, TableContainer, 
  TableHead, TableRow, Paper, IconButton 
} from '@mui/material';
import { Add, Edit, Delete, PlayArrow, Stop } from '@mui/icons-material';
import axios from '../utils/axios';

function Algorithms() {
  const [algorithms, setAlgorithms] = useState([]);
  const [models, setModels] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [openDialog, setOpenDialog] = useState(false);
  const [editingAlgorithm, setEditingAlgorithm] = useState(null);
  const [formData, setFormData] = useState({
    modelId: '',
    cameraId: '',
    confidence: 0.5,
    alertThreshold: 3,
    regions: [],
    notificationEnabled: true
  });
  const [runningAlgorithms, setRunningAlgorithms] = useState(new Set());

  useEffect(() => {
    fetchAll();
  }, []);

  const fetchAll = async () => {
    try {
      const [algorithmsRes, modelsRes, camerasRes] = await Promise.all([
        axios.get('/api/algorithms'),
        axios.get('/api/models'),
        axios.get('/api/cameras')
      ]);
      setAlgorithms(algorithmsRes || []);
      setModels(modelsRes || []);
      setCameras(camerasRes || []);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const handleCreate = async () => {
    try {
      await axios.post('/api/algorithms', formData);
      setOpenDialog(false);
      resetForm();
      fetchAll();
    } catch (error) {
      console.error('Error creating algorithm:', error);
    }
  };

  const handleUpdate = async () => {
    try {
      await axios.put(`/api/algorithms/${editingAlgorithm.id}`, formData);
      setOpenDialog(false);
      resetForm();
      fetchAll();
    } catch (error) {
      console.error('Error updating algorithm:', error);
    }
  };

  const handleDelete = async (id) => {
    try {
      await axios.delete(`/api/algorithms/${id}`);
      fetchAll();
    } catch (error) {
      console.error('Error deleting algorithm:', error);
    }
  };

  const handleEdit = (algorithm) => {
    setEditingAlgorithm(algorithm);
    setFormData({
      modelId: algorithm.modelId || '',
      cameraId: algorithm.cameraId || '',
      confidence: algorithm.confidence || 0.5,
      alertThreshold: algorithm.alertThreshold || 3,
      regions: algorithm.regions || [],
      notificationEnabled: algorithm.notificationEnabled
    });
    setOpenDialog(true);
  };

  const resetForm = () => {
    setEditingAlgorithm(null);
    setFormData({
      modelId: '',
      cameraId: '',
      confidence: 0.5,
      alertThreshold: 3,
      regions: [],
      notificationEnabled: true
    });
  };

  const handleStartDetection = async (algorithm) => {
    try {
      await axios.post('/api/detection/start', {
        camera_id: algorithm.cameraId,
        model_id: algorithm.modelId,
        settings: {
          confidence: algorithm.confidence,
          alert_threshold: algorithm.alertThreshold,
          regions: algorithm.regions,
          notification_enabled: algorithm.notificationEnabled
        }
      });
      setRunningAlgorithms(prev => new Set([...prev, algorithm.id]));
    } catch (error) {
      console.error('Error starting detection:', error);
    }
  };

  const handleStopDetection = async (algorithm) => {
    try {
      await axios.post('/api/detection/stop', {
        camera_id: algorithm.cameraId
      });
      setRunningAlgorithms(prev => {
        const next = new Set(prev);
        next.delete(algorithm.id);
        return next;
      });
    } catch (error) {
      console.error('Error stopping detection:', error);
    }
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => {
            resetForm();
            setOpenDialog(true);
          }}
          sx={{ mb: 2 }}
        >
          添加任务
        </Button>

        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>ID</TableCell>
                <TableCell>模型</TableCell>
                <TableCell>摄像头</TableCell>
                <TableCell>置信度</TableCell>
                <TableCell>告警阈值</TableCell>
                <TableCell>状态</TableCell>
                <TableCell>操作</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {algorithms.map((algorithm) => (
                <TableRow key={algorithm.id}>
                  <TableCell>{algorithm.id}</TableCell>
                  <TableCell>
                    {models.find(m => m.id === algorithm.modelId)?.name || '-'}
                  </TableCell>
                  <TableCell>
                    {cameras.find(c => c.id === algorithm.cameraId)?.name || '-'}
                  </TableCell>
                  <TableCell>{algorithm.confidence}</TableCell>
                  <TableCell>{algorithm.alertThreshold}</TableCell>
                  <TableCell>
                    {runningAlgorithms.has(algorithm.id) ? '运行中' : '已停止'}
                  </TableCell>
                  <TableCell>
                    <IconButton
                      color="primary"
                      onClick={() => handleEdit(algorithm)}
                      title="编辑"
                      disabled={runningAlgorithms.has(algorithm.id)}
                    >
                      <Edit />
                    </IconButton>
                    <IconButton
                      color="error"
                      onClick={() => handleDelete(algorithm.id)}
                      title="删除"
                      disabled={runningAlgorithms.has(algorithm.id)}
                    >
                      <Delete />
                    </IconButton>
                    {!runningAlgorithms.has(algorithm.id) ? (
                      <IconButton
                        color="success"
                        onClick={() => handleStartDetection(algorithm)}
                        title="开始推理"
                      >
                        <PlayArrow />
                      </IconButton>
                    ) : (
                      <IconButton
                        color="warning"
                        onClick={() => handleStopDetection(algorithm)}
                        title="停止推理"
                      >
                        <Stop />
                      </IconButton>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Grid>

      <Dialog 
        open={openDialog} 
        onClose={() => setOpenDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {editingAlgorithm ? '编辑任务' : '添加任务'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>选择模型</InputLabel>
                <Select
                  value={formData.modelId}
                  onChange={(e) => setFormData({ ...formData, modelId: e.target.value })}
                >
                  {models.map((model) => (
                    <MenuItem key={model.id} value={model.id}>
                      {model.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>选择摄像头</InputLabel>
                <Select
                  value={formData.cameraId}
                  onChange={(e) => setFormData({ ...formData, cameraId: e.target.value })}
                >
                  {cameras.map((camera) => (
                    <MenuItem key={camera.id} value={camera.id}>
                      {camera.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="置信度阈值"
                type="number"
                value={formData.confidence}
                onChange={(e) => setFormData({ ...formData, confidence: parseFloat(e.target.value) })}
                inputProps={{ step: 0.1, min: 0, max: 1 }}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="告警阈值"
                type="number"
                value={formData.alertThreshold}
                onChange={(e) => setFormData({ ...formData, alertThreshold: parseInt(e.target.value) })}
                inputProps={{ min: 1 }}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>取消</Button>
          <Button 
            onClick={editingAlgorithm ? handleUpdate : handleCreate}
            color="primary"
          >
            {editingAlgorithm ? '更新' : '添加'}
          </Button>
        </DialogActions>
      </Dialog>
    </Grid>
  );
}

export default Algorithms; 