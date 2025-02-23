import React, { useState, useEffect } from 'react';
import {
  Grid, Paper, TextField, Button, Typography, Snackbar, Alert,
  Card, CardContent, CardActions, Divider
} from '@mui/material';
import { Save } from '@mui/icons-material';
import axios from '../utils/axios';

function Settings() {
  const [settings, setSettings] = useState({
    external_api: {
      url: '',
      token: '',
      secret: ''
    },
    alert: {
      retention_days: 30,
      image_quality: 95
    },
    system: {
      log_level: 'INFO'
    }
  });
  const [message, setMessage] = useState({ type: '', content: '' });
  const [openSnackbar, setOpenSnackbar] = useState(false);

  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      const response = await axios.get('/api/settings');
      setSettings(response.data);
    } catch (error) {
      setMessage({
        type: 'error',
        content: '获取配置失败: ' + error.message
      });
      setOpenSnackbar(true);
    }
  };

  const handleSave = async () => {
    try {
      await axios.post('/api/settings', settings);
      setMessage({
        type: 'success',
        content: '保存成功'
      });
      setOpenSnackbar(true);
    } catch (error) {
      setMessage({
        type: 'error',
        content: '保存失败: ' + error.message
      });
      setOpenSnackbar(true);
    }
  };

  const handleChange = (section, field) => (event) => {
    setSettings(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: event.target.value
      }
    }));
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h5" gutterBottom>系统设置</Typography>
      </Grid>

      {/* 外部API设置 */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>外部API设置</Typography>
            <TextField
              fullWidth
              label="API地址"
              value={settings.external_api.url}
              onChange={handleChange('external_api', 'url')}
              margin="normal"
            />
            <TextField
              fullWidth
              label="Token"
              value={settings.external_api.token}
              onChange={handleChange('external_api', 'token')}
              margin="normal"
            />
            <TextField
              fullWidth
              label="Secret"
              type="password"
              value={settings.external_api.secret}
              onChange={handleChange('external_api', 'secret')}
              margin="normal"
            />
          </CardContent>
        </Card>
      </Grid>

      {/* 告警设置 */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>告警设置</Typography>
            <TextField
              fullWidth
              type="number"
              label="告警保留天数"
              value={settings.alert.retention_days}
              onChange={handleChange('alert', 'retention_days')}
              margin="normal"
            />
            <TextField
              fullWidth
              type="number"
              label="图片质量(1-100)"
              value={settings.alert.image_quality}
              onChange={handleChange('alert', 'image_quality')}
              margin="normal"
            />
          </CardContent>
        </Card>
      </Grid>

      {/* 系统设置 */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>系统设置</Typography>
            <TextField
              fullWidth
              select
              label="日志级别"
              value={settings.system.log_level}
              onChange={handleChange('system', 'log_level')}
              margin="normal"
              SelectProps={{
                native: true
              }}
            >
              <option value="DEBUG">DEBUG</option>
              <option value="INFO">INFO</option>
              <option value="WARNING">WARNING</option>
              <option value="ERROR">ERROR</option>
            </TextField>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12}>
        <Button
          variant="contained"
          color="primary"
          startIcon={<Save />}
          onClick={handleSave}
        >
          保存设置
        </Button>
      </Grid>

      <Snackbar
        open={openSnackbar}
        autoHideDuration={3000}
        onClose={() => setOpenSnackbar(false)}
      >
        <Alert severity={message.type} onClose={() => setOpenSnackbar(false)}>
          {message.content}
        </Alert>
      </Snackbar>
    </Grid>
  );
}

export default Settings; 