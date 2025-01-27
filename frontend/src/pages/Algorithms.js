import React, { useState, useEffect } from 'react';
import {
  Grid, Paper, Table, TableBody, TableCell, TableContainer, TableHead, 
  TableRow, Button, Dialog, DialogTitle, DialogContent, DialogActions,
  TextField, FormControl, InputLabel, Select, MenuItem
} from '@mui/material';
import { Add } from '@mui/icons-material';
import axios from '../utils/axios';

function Algorithms() {
  const [algorithms, setAlgorithms] = useState([]);
  const [openDialog, setOpenDialog] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    type: 'object_detection',
    description: '',
    parameters: {}
  });

  useEffect(() => {
    fetchAlgorithms();
  }, []);

  const fetchAlgorithms = async () => {
    try {
      const response = await axios.get('/api/algorithms');
      setAlgorithms(response || []);
    } catch (error) {
      console.error('Error fetching algorithms:', error);
    }
  };

  const handleCreate = async () => {
    try {
      await axios.post('/api/algorithms', formData);
      setOpenDialog(false);
      resetForm();
      fetchAlgorithms();
    } catch (error) {
      console.error('Error creating algorithm:', error);
    }
  };

  const resetForm = () => {
    setFormData({
      name: '',
      type: 'object_detection',
      description: '',
      parameters: {}
    });
  };

  const algorithmTypes = [
    { value: 'object_detection', label: '目标检测' },
    { value: 'behavior_analysis', label: '行为分析' }
  ];

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
          添加算法
        </Button>

        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>ID</TableCell>
                <TableCell>名称</TableCell>
                <TableCell>类型</TableCell>
                <TableCell>描述</TableCell>
                <TableCell>创建时间</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {algorithms.map((algorithm) => (
                <TableRow key={algorithm.id}>
                  <TableCell>{algorithm.id}</TableCell>
                  <TableCell>{algorithm.name}</TableCell>
                  <TableCell>
                    {algorithmTypes.find(t => t.value === algorithm.type)?.label || algorithm.type}
                  </TableCell>
                  <TableCell>{algorithm.description}</TableCell>
                  <TableCell>{new Date(algorithm.created_at).toLocaleString()}</TableCell>
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
        <DialogTitle>添加算法</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="算法名称"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>算法类型</InputLabel>
                <Select
                  value={formData.type}
                  onChange={(e) => setFormData({ ...formData, type: e.target.value })}
                >
                  {algorithmTypes.map((type) => (
                    <MenuItem key={type.value} value={type.value}>
                      {type.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={4}
                label="算法描述"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>取消</Button>
          <Button onClick={handleCreate} color="primary">
            添加
          </Button>
        </DialogActions>
      </Dialog>
    </Grid>
  );
}

export default Algorithms; 