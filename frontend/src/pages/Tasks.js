import React, { useState, useEffect } from 'react';
import {
  Grid, Paper, Table, TableBody, TableCell, TableContainer, TableHead,
  TableRow, Button, Dialog, DialogTitle, DialogContent, DialogActions,
  TextField, Switch, FormControlLabel, Select, MenuItem, IconButton,
  Typography
} from '@mui/material';
import { Add, Edit, Delete, PlayArrow, Stop } from '@mui/icons-material';
import axios from '../utils/axios';
import BeltCalibrationTool from '../components/BeltCalibrationTool';
import BeltDeviationCalibrationTool from '../components/BeltDeviationCalibrationTool';

function Tasks() {
  const [tasks, setTasks] = useState([]);
  const [models, setModels] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [algorithms, setAlgorithms] = useState([]);
  const [openDialog, setOpenDialog] = useState(false);
  const [editingTask, setEditingTask] = useState(null);
  const [formData, setFormData] = useState({
    name: '',
    modelId: '',
    cameraId: '',
    confidence: 0.5,
    alertThreshold: 3,
    notificationEnabled: true,
    algorithm_id: '',
    algorithm_parameters: {
      min_area_cm2: 100,
      calibration: {
        belt_width: 0,
        points: []
      },
      regions: []
    }
  });
  const [runningTasks, setRunningTasks] = useState(new Set());

  useEffect(() => {
    fetchAll();
  }, []);

  const fetchAll = async () => {
    try {
      const [tasksRes, modelsRes, camerasRes, algorithmsRes] = await Promise.all([
        axios.get('/api/tasks'),
        axios.get('/api/models'),
        axios.get('/api/cameras'),
        axios.get('/api/algorithms')
      ]);
      setTasks(tasksRes || []);
      setModels(modelsRes || []);
      setCameras(camerasRes || []);
      setAlgorithms(algorithmsRes || []);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const handleCreate = async () => {
    try {
      const response = await axios.post('/api/tasks', formData);
      setTasks([...tasks, response]);
      setOpenDialog(false);
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
      name: task.name,
      modelId: task.modelId,
      cameraId: task.cameraId,
      confidence: task.confidence,
      alertThreshold: task.alertThreshold,
      notificationEnabled: task.notificationEnabled,
      algorithm_id: task.algorithm_id,
      algorithm_parameters: task.algorithm_parameters || {
        min_area_cm2: 100,
        calibration: {
          belt_width: 0,
          points: []
        },
        regions: []
      }
    });
    setOpenDialog(true);
  };

  const resetForm = () => {
    setEditingTask(null);
    setFormData({
      name: '',
      modelId: '',
      cameraId: '',
      confidence: 0.5,
      alertThreshold: 3,
      notificationEnabled: true,
      algorithm_id: '',
      algorithm_parameters: {
        min_area_cm2: 100,
        calibration: {
          belt_width: 0,
          points: []
        },
        regions: []
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
          algorithm_id: task.algorithm_id,
          algorithm_parameters: task.algorithm_parameters
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
    setFormData({
      ...formData,
      algorithm_parameters: {
        ...formData.algorithm_parameters,
        calibration: calibrationData.calibration  // 直接传递所有标定数据
      }
    });
  };

  // 根据算法类型返回对应的标定工具
  const renderCalibrationTool = () => {
    if (!formData.algorithm_id) return null;

    const algorithm = algorithms.find(a => a.id === formData.algorithm_id);
    if (!algorithm) return null;

    switch (algorithm.name) {
      case '皮带破损检测':
        return (
          <BeltCalibrationTool 
            cameraId={formData.cameraId}
            onCalibrate={handleCalibrate}
          />
        );
      case '皮带跑偏检测':
        return (
          <BeltDeviationCalibrationTool
            cameraId={formData.cameraId}
            onCalibrate={handleCalibrate}
          />
        );
      default:
        return null;
    }
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <Typography variant="h5">任务列表</Typography>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => {
              resetForm();
              setOpenDialog(true);
            }}
          >
            添加任务
          </Button>
        </div>
      </Grid>

      <Grid item xs={12}>
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>名称</TableCell>
                <TableCell>视频源</TableCell>
                <TableCell>模型</TableCell>
                <TableCell>算法</TableCell>
                <TableCell>状态</TableCell>
                <TableCell>操作</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {tasks.map((task) => (
                <TableRow key={task.id}>
                  <TableCell>{task.name}</TableCell>
                  <TableCell>{cameras.find(c => c.id === task.cameraId)?.name}</TableCell>
                  <TableCell>{models.find(m => m.id === task.modelId)?.name}</TableCell>
                  <TableCell>{algorithms.find(a => a.id === task.algorithm_id)?.name}</TableCell>
                  <TableCell>{task.status}</TableCell>
                  <TableCell>
                    <IconButton onClick={() => handleEdit(task)}>
                      <Edit />
                    </IconButton>
                    <IconButton onClick={() => handleDelete(task.id)}>
                      <Delete />
                    </IconButton>
                    {runningTasks.has(task.id) ? (
                      <IconButton onClick={() => handleStopDetection(task)}>
                        <Stop />
                      </IconButton>
                    ) : (
                      <IconButton onClick={() => handleStartDetection(task)}>
                        <PlayArrow />
                      </IconButton>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Grid>

      <Dialog open={openDialog} onClose={() => setOpenDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          {editingTask ? '编辑任务' : '创建任务'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="任务名称"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                margin="normal"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Select
                fullWidth
                value={formData.modelId}
                onChange={(e) => setFormData({ ...formData, modelId: e.target.value })}
                displayEmpty
              >
                <MenuItem value="">选择模型</MenuItem>
                {models.map(model => (
                  <MenuItem key={model.id} value={model.id}>
                    {model.name}
                  </MenuItem>
                ))}
              </Select>
            </Grid>

            <Grid item xs={12} md={6}>
              <Select
                fullWidth
                value={formData.cameraId}
                onChange={(e) => setFormData({ ...formData, cameraId: e.target.value })}
                displayEmpty
              >
                <MenuItem value="">选择视频源</MenuItem>
                {cameras.map(camera => (
                  <MenuItem key={camera.id} value={camera.id}>
                    {camera.name}
                  </MenuItem>
                ))}
              </Select>
            </Grid>

            <Grid item xs={12} md={6}>
              <Select
                fullWidth
                value={formData.algorithm_id}
                onChange={(e) => setFormData({ ...formData, algorithm_id: e.target.value })}
                displayEmpty
              >
                <MenuItem value="">选择算法</MenuItem>
                {algorithms.map(algorithm => (
                  <MenuItem key={algorithm.id} value={algorithm.id}>
                    {algorithm.name}
                  </MenuItem>
                ))}
              </Select>
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                label="置信度"
                value={formData.confidence}
                onChange={(e) => setFormData({ ...formData, confidence: parseFloat(e.target.value) })}
                inputProps={{ step: 0.1, min: 0, max: 1 }}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                label="告警间隔(秒)"
                value={formData.alertThreshold}
                onChange={(e) => setFormData({ ...formData, alertThreshold: parseInt(e.target.value) })}
                inputProps={{ min: 1 }}
              />
            </Grid>

            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.notificationEnabled}
                    onChange={(e) => setFormData({ ...formData, notificationEnabled: e.target.checked })}
                  />
                }
                label="启用通知"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                label="最小异常面积(cm²)"
                value={formData.algorithm_parameters.min_area_cm2}
                onChange={(e) => setFormData({
                  ...formData,
                  algorithm_parameters: {
                    ...formData.algorithm_parameters,
                    min_area_cm2: parseInt(e.target.value)
                  }
                })}
                inputProps={{ min: 1 }}
              />
            </Grid>

            <Grid item xs={12}>
              {renderCalibrationTool()}
            </Grid>
            
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>取消</Button>
          <Button onClick={editingTask ? handleUpdate : handleCreate} variant="contained">
            {editingTask ? '更新' : '创建'}
          </Button>
        </DialogActions>
      </Dialog>
    </Grid>
  );
}

export default Tasks; 