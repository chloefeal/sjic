import React, { useState, useEffect } from 'react';
import { 
  Grid, Card, CardContent, Typography, Button, Dialog, DialogTitle, 
  DialogContent, DialogActions, TextField, FormControl, InputLabel, 
  Select, MenuItem, Table, TableBody, TableCell, TableContainer, 
  TableHead, TableRow, Paper, IconButton 
} from '@mui/material';
import { Add, Edit, Delete, PlayArrow, Stop } from '@mui/icons-material';
import axios from '../utils/axios';
import CalibrationTool from '../components/CalibrationTool';

function Tasks() {
  const [tasks, setTasks] = useState([]);
  const [models, setModels] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [openDialog, setOpenDialog] = useState(false);
  const [editingTask, setEditingTask] = useState(null);
  const [formData, setFormData] = useState({
    modelId: '',
    cameraId: '',
    id: '',
    confidence: 0.5,
    alertThreshold: 3,
    regions: [],
    notificationEnabled: true,
    parameters: {
      pixel_to_cm: 0.1,
      calibration: {
        belt_width: 0,
        points: []
      }
    }
  });
  const [runningTasks, setRunningTasks] = useState(new Set());

  useEffect(() => {
    fetchAll();
  }, []);

  const fetchAll = async () => {
    try {
      const [tasksRes, modelsRes, camerasRes] = await Promise.all([
        axios.get('/api/tasks'),
        axios.get('/api/models'),
        axios.get('/api/cameras')
      ]);
      setTasks(tasksRes || []);
      setModels(modelsRes || []);
      setCameras(camerasRes || []);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const handleCreate = async () => {
    try {
      await axios.post('/api/tasks', formData);
      setOpenDialog(false);
      resetForm();
      fetchAll();
    } catch (error) {
      console.error('Error creating task:', error);
    }
  };

  const handleUpdate = async () => {
    try {
      await axios.put(`/api/tasks/${editingTask.id}`, formData);
      setOpenDialog(false);
      resetForm();
      fetchAll();
    } catch (error) {
      console.error('Error updating task:', error);
    }
  };

  const handleDelete = async (id) => {
    try {
      await axios.delete(`/api/tasks/${id}`);
      fetchAll();
    } catch (error) {
      console.error('Error deleting task:', error);
    }
  };

  const handleEdit = (task) => {
    setEditingTask(task);
    setFormData({
      modelId: task.modelId || '',
      cameraId: task.cameraId || '',
      confidence: task.confidence || 0.5,
      alertThreshold: task.alertThreshold || 3,
      regions: task.regions || [],
      notificationEnabled: task.notificationEnabled,
      parameters: task.parameters || {
        pixel_to_cm: 0.1,
        calibration: {
          belt_width: 0,
          points: []
        }
      }
    });
    setOpenDialog(true);
  };

  const resetForm = () => {
    setEditingTask(null);
    setFormData({
      modelId: '',
      cameraId: '',
      confidence: 0.5,
      alertThreshold: 3,
      regions: [],
      notificationEnabled: true,
      parameters: {
        pixel_to_cm: 0.1,
        calibration: {
          belt_width: 0,
          points: []
        }
      }
    });
  };

  const handleStartDetection = async (task) => {
    try {
      await axios.post('/api/detection/start', {
        camera_id: task.cameraId,
        model_id: task.modelId,
        task_id: task.id,
        settings: {
          confidence: task.confidence,
          alert_threshold: task.alertThreshold,
          regions: task.regions,
          notification_enabled: task.notificationEnabled,
          pixel_to_cm: task.parameters.pixel_to_cm,
          calibration: task.parameters.calibration
        }
      });
      setRunningTasks(prev => new Set([...prev, task.id]));
    } catch (error) {
      console.error('Error starting detection:', error);
    }
  };

  const handleStopDetection = async (task) => {
    try {
      await axios.post('/api/detection/stop', {
        task_id: task.id
      });
      setRunningTasks(prev => {
        const next = new Set(prev);
        next.delete(task.id);
        return next;
      });
    } catch (error) {
      console.error('Error stopping detection:', error);
    }
  };

  const handleCalibrate = (calibrationData) => {
    setFormData(prev => ({
      ...prev,
      parameters: {
        ...prev.parameters,
        pixel_to_cm: calibrationData.pixel_to_cm,
        calibration: calibrationData.calibration
      }
    }));
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
                <TableCell>名称</TableCell>
                <TableCell>模型</TableCell>
                <TableCell>摄像头</TableCell>
                <TableCell>置信度</TableCell>
                <TableCell>告警阈值</TableCell>
                <TableCell>状态</TableCell>
                <TableCell>操作</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {tasks.map((task) => (
                <TableRow key={task.id}>
                  <TableCell>{task.id}</TableCell>
                  <TableCell>{task.name}</TableCell>
                  <TableCell>
                    {models.find(m => m.id === task.modelId)?.name || '-'}
                  </TableCell>
                  <TableCell>
                    {cameras.find(c => c.id === task.cameraId)?.name || '-'}
                  </TableCell>
                  <TableCell>{task.confidence}</TableCell>
                  <TableCell>{task.alertThreshold}</TableCell>
                  <TableCell>
                    {runningTasks.has(task.id) ? '运行中' : '已停止'}
                  </TableCell>
                  <TableCell>
                    <IconButton
                      color="primary"
                      onClick={() => handleEdit(task)}
                      title="编辑"
                      disabled={runningTasks.has(task.id)}
                    >
                      <Edit />
                    </IconButton>
                    <IconButton
                      color="error"
                      onClick={() => handleDelete(task.id)}
                      title="删除"
                      disabled={runningTasks.has(task.id)}
                    >
                      <Delete />
                    </IconButton>
                    {!runningTasks.has(task.id) ? (
                      <IconButton
                        color="success"
                        onClick={() => handleStartDetection(task)}
                        title="开始推理"
                      >
                        <PlayArrow />
                      </IconButton>
                    ) : (
                      <IconButton
                        color="warning"
                        onClick={() => handleStopDetection(task)}
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
          {editingTask ? '编辑任务' : '添加任务'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="名称"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
            </Grid>
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
            <Grid item xs={12}>
              <CalibrationTool onCalibrate={handleCalibrate} />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>取消</Button>
          <Button 
            onClick={editingTask ? handleUpdate : handleCreate}
            color="primary"
          >
            {editingTask ? '更新' : '添加'}
          </Button>
        </DialogActions>
      </Dialog>
    </Grid>
  );
}

export default Tasks; 