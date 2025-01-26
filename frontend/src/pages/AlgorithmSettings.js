import React, { useState, useEffect } from 'react';
import { 
  Grid, Card, CardContent, Typography, Button, Dialog, DialogTitle, 
  DialogContent, DialogActions, TextField, FormControl, InputLabel, 
  Select, MenuItem, Table, TableBody, TableCell, TableContainer, 
  TableHead, TableRow, Paper, IconButton 
} from '@mui/material';
import { Add, Edit, Delete } from '@mui/icons-material';
import axios from '../utils/axios';

function AlgorithmSettings() {
  const [settings, setSettings] = useState([]);
  const [models, setModels] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [openDialog, setOpenDialog] = useState(false);
  const [editingSettings, setEditingSettings] = useState(null);
  const [formData, setFormData] = useState({
    modelId: '',
    cameraId: '',
    confidence: 0.5,
    alertThreshold: 3,
    regions: [],
    notificationEnabled: true
  });

  useEffect(() => {
    fetchAll();
  }, []);

  const fetchAll = async () => {
    try {
      const [settingsRes, modelsRes, camerasRes] = await Promise.all([
        axios.get('/api/settings'),
        axios.get('/api/models'),
        axios.get('/api/cameras')
      ]);
      setSettings(settingsRes || []);
      setModels(modelsRes || []);
      setCameras(camerasRes || []);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const handleCreate = async () => {
    try {
      await axios.post('/api/settings', formData);
      setOpenDialog(false);
      resetForm();
      fetchAll();
    } catch (error) {
      console.error('Error creating settings:', error);
    }
  };

  const handleUpdate = async () => {
    try {
      await axios.put(`/api/settings/${editingSettings.id}`, formData);
      setOpenDialog(false);
      resetForm();
      fetchAll();
    } catch (error) {
      console.error('Error updating settings:', error);
    }
  };

  const handleDelete = async (id) => {
    try {
      await axios.delete(`/api/settings/${id}`);
      fetchAll();
    } catch (error) {
      console.error('Error deleting settings:', error);
    }
  };

  const handleEdit = (setting) => {
    setEditingSettings(setting);
    setFormData({
      modelId: setting.modelId || '',
      cameraId: setting.cameraId || '',
      confidence: setting.confidence || 0.5,
      alertThreshold: setting.alertThreshold || 3,
      regions: setting.regions || [],
      notificationEnabled: setting.notificationEnabled
    });
    setOpenDialog(true);
  };

  const resetForm = () => {
    setEditingSettings(null);
    setFormData({
      modelId: '',
      cameraId: '',
      confidence: 0.5,
      alertThreshold: 3,
      regions: [],
      notificationEnabled: true
    });
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
          添加设置
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
                <TableCell>操作</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {settings.map((setting) => (
                <TableRow key={setting.id}>
                  <TableCell>{setting.id}</TableCell>
                  <TableCell>
                    {models.find(m => m.id === setting.modelId)?.name || '-'}
                  </TableCell>
                  <TableCell>
                    {cameras.find(c => c.id === setting.cameraId)?.name || '-'}
                  </TableCell>
                  <TableCell>{setting.confidence}</TableCell>
                  <TableCell>{setting.alertThreshold}</TableCell>
                  <TableCell>
                    <IconButton
                      color="primary"
                      onClick={() => handleEdit(setting)}
                      title="编辑"
                    >
                      <Edit />
                    </IconButton>
                    <IconButton
                      color="error"
                      onClick={() => handleDelete(setting.id)}
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

      <Dialog 
        open={openDialog} 
        onClose={() => setOpenDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {editingSettings ? '编辑设置' : '添加设置'}
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
            onClick={editingSettings ? handleUpdate : handleCreate}
            color="primary"
          >
            {editingSettings ? '更新' : '添加'}
          </Button>
        </DialogActions>
      </Dialog>
    </Grid>
  );
}

export default AlgorithmSettings; 