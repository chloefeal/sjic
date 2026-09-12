import React, { useState, useEffect, useCallback } from 'react';
import {
  Grid, TextField, Button, Typography, Snackbar, Alert,
  Card, CardContent, Box, Avatar, Stack, Chip, Divider
} from '@mui/material';
import { Save, CloudUpload, Delete, RestartAlt, ContentCopy, VpnKey } from '@mui/icons-material';
import axios, { getBaseUrl } from '../utils/axios';

const DEFAULT_SETTINGS = {
  external_alert_api: {
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
  },
  branding: {
    company_name: '',
    product_name: '智算检测平台',
    logo_filename: '',
    logo_url: ''
  }
};

const EDITION_LABEL = {
  trial: '试用版',
  official: '正式版',
};

function Settings() {
  const isVendor = localStorage.getItem('user_role') === 'vendor';
  const isCustomerAdmin = localStorage.getItem('user_role') === 'customer';
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [license, setLicense] = useState(null);
  const [message, setMessage] = useState({ type: '', content: '' });
  const [openSnackbar, setOpenSnackbar] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [importingLicense, setImportingLicense] = useState(false);
  const [restartingBackend, setRestartingBackend] = useState(false);

  const showMsg = (type, content) => {
    setMessage({ type, content });
    setOpenSnackbar(true);
  };

  const fetchLicense = useCallback(async () => {
    try {
      const status = await axios.get('/api/license/status');
      setLicense(status);
    } catch (error) {
      showMsg('error', '获取授权状态失败: ' + (error.response?.data?.error || error.message));
    }
  }, []);

  const fetchSettings = useCallback(async () => {
    try {
      const response = await axios.get('/api/settings');
      const mergedSettings = {
        ...DEFAULT_SETTINGS,
        ...response,
        external_alert_api: {
          ...DEFAULT_SETTINGS.external_alert_api,
          ...(response.external_alert_api || {})
        },
        alert: {
          ...DEFAULT_SETTINGS.alert,
          ...(response.alert || {})
        },
        system: {
          ...DEFAULT_SETTINGS.system,
          ...(response.system || {})
        },
        branding: {
          ...DEFAULT_SETTINGS.branding,
          ...(response.branding || {})
        }
      };
      setSettings(mergedSettings);
    } catch (error) {
      // 未授权时业务 API 会 403，授权卡片仍可用
      if (error.response?.status !== 403) {
        showMsg('error', '获取配置失败: ' + error.message);
      }
    }
  }, []);

  useEffect(() => {
    fetchLicense();
    fetchSettings();
  }, [fetchLicense, fetchSettings]);

  const handleSave = async () => {
    try {
      const payload = {
        external_alert_api: settings.external_alert_api,
        alert: settings.alert,
        system: settings.system,
      };
      if (isVendor) {
        payload.branding = {
          company_name: settings.branding.company_name,
          product_name: settings.branding.product_name,
          logo_filename: settings.branding.logo_filename || '',
        };
      }
      await axios.post('/api/settings', payload);
      showMsg('success', '保存成功');
      window.dispatchEvent(new Event('branding-updated'));
    } catch (error) {
      showMsg('error', '保存失败: ' + (error.response?.data?.error || error.message));
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

  const handleLogoUpload = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('logo', file);
      const result = await axios.post('/api/settings/logo', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setSettings(prev => ({
        ...prev,
        branding: {
          ...prev.branding,
          ...(result.branding || {}),
        }
      }));
      showMsg('success', 'Logo 上传成功');
      window.dispatchEvent(new Event('branding-updated'));
    } catch (error) {
      showMsg('error', 'Logo 上传失败: ' + (error.response?.data?.error || error.message));
    } finally {
      setUploading(false);
    }
  };

  const handleLogoRemove = async () => {
    try {
      const result = await axios.delete('/api/settings/logo');
      setSettings(prev => ({
        ...prev,
        branding: {
          ...prev.branding,
          ...(result.branding || {}),
          logo_filename: '',
          logo_url: '',
        }
      }));
      showMsg('success', 'Logo 已清除');
      window.dispatchEvent(new Event('branding-updated'));
    } catch (error) {
      showMsg('error', '清除失败: ' + error.message);
    }
  };

  const handleRestartBackend = async () => {
    if (!window.confirm('确定远程重启管理平台后端？服务将短暂中断，Docker 部署下会自动拉起。')) {
      return;
    }
    setRestartingBackend(true);
    try {
      const result = await axios.post('/api/admin/restart-backend');
      showMsg('success', result.message || '后端正在重启…');
    } catch (error) {
      showMsg('error', '重启失败: ' + (error.response?.data?.error || error.message));
      setRestartingBackend(false);
    }
  };

  const handleCopyMachineCode = async () => {
    const code = license?.machine_code || '';
    if (!code) return;
    try {
      await navigator.clipboard.writeText(code);
      showMsg('success', '机器码已复制');
    } catch (e) {
      showMsg('error', '复制失败，请手动选择机器码');
    }
  };

  const handleLicenseImport = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    setImportingLicense(true);
    try {
      const formData = new FormData();
      formData.append('license', file);
      const result = await axios.post('/api/license/import', formData);
      setLicense(result.status || result);
      showMsg('success', result.message || '授权导入成功');
      window.dispatchEvent(new Event('license-updated'));
      fetchSettings();
    } catch (error) {
      showMsg('error', '导入失败: ' + (error.response?.data?.error || error.message));
      if (error.response?.data?.status) {
        setLicense(error.response.data.status);
      }
    } finally {
      setImportingLicense(false);
    }
  };

  const logoSrc = settings.branding.logo_url
    ? `${getBaseUrl()}${settings.branding.logo_url}`
    : '';

  const licenseValid = Boolean(license?.valid);
  const editionLabel = EDITION_LABEL[license?.edition] || (license?.edition ? license.edition : '未授权');

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h5" gutterBottom>系统设置</Typography>
      </Grid>

      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
              <VpnKey fontSize="small" />
              <Typography variant="h6">授权管理</Typography>
              <Chip
                size="small"
                label={licenseValid ? '有效' : '无效 / 未导入'}
                color={licenseValid ? 'success' : 'warning'}
              />
              <Chip size="small" label={editionLabel} variant="outlined" />
            </Stack>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              新部署需先导入试用版或正式版授权后才能使用业务功能。试用版有效期一个月，授权控制可接入视频路数。
              请将下方机器码发给发行方以生成绑定本机的授权文件。
            </Typography>

            <TextField
              fullWidth
              label="本机机器码"
              value={license?.machine_code || ''}
              margin="normal"
              InputProps={{ readOnly: true }}
              helperText="绑定宿主机/虚机标识，克隆镜像后通常会变化"
            />
            <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
              <Button
                variant="outlined"
                startIcon={<ContentCopy />}
                onClick={handleCopyMachineCode}
                disabled={!license?.machine_code}
              >
                复制机器码
              </Button>
              <Button
                variant="contained"
                component="label"
                startIcon={<CloudUpload />}
                disabled={importingLicense}
              >
                {importingLicense ? '导入中…' : '导入授权文件'}
                <input type="file" hidden accept=".json,application/json" onChange={handleLicenseImport} />
              </Button>
            </Stack>

            <Divider sx={{ my: 1.5 }} />
            <Grid container spacing={1}>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="caption" color="text.secondary">视频路数上限</Typography>
                <Typography>
                  {licenseValid
                    ? (license.max_cameras > 0 ? license.max_cameras : '不限')
                    : '—'}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="caption" color="text.secondary">到期时间</Typography>
                <Typography>
                  {license?.expires_at || '—'}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="caption" color="text.secondary">客户标识</Typography>
                <Typography>{license?.customer_id || '—'}</Typography>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="caption" color="text.secondary">状态说明</Typography>
                <Typography>{license?.message || (licenseValid ? '正常' : '—')}</Typography>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Grid>

      {isVendor && (
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>品牌定制</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                用于集成商 / 运营商白标：登录页与顶栏将显示公司名称与 Logo。
              </Typography>
              <TextField
                fullWidth
                label="公司名称"
                value={settings.branding.company_name}
                onChange={handleChange('branding', 'company_name')}
                margin="normal"
                disabled={!licenseValid}
              />
              <TextField
                fullWidth
                label="产品名称"
                value={settings.branding.product_name}
                onChange={handleChange('branding', 'product_name')}
                margin="normal"
                disabled={!licenseValid}
              />
              <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: 2 }}>
                <Avatar
                  variant="rounded"
                  src={logoSrc || undefined}
                  sx={{ width: 64, height: 64, bgcolor: 'grey.800' }}
                >
                  {!logoSrc && 'Logo'}
                </Avatar>
                <Box>
                  <Button
                    variant="outlined"
                    component="label"
                    startIcon={<CloudUpload />}
                    disabled={uploading || !licenseValid}
                    sx={{ mr: 1 }}
                  >
                    {uploading ? '上传中…' : '上传 Logo'}
                    <input type="file" hidden accept="image/*" onChange={handleLogoUpload} />
                  </Button>
                  {logoSrc && (
                    <Button
                      color="inherit"
                      startIcon={<Delete />}
                      onClick={handleLogoRemove}
                      disabled={!licenseValid}
                    >
                      清除
                    </Button>
                  )}
                </Box>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
      )}

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>外部告警API设置</Typography>
            <TextField
              fullWidth
              label="API地址"
              value={settings.external_alert_api.url}
              onChange={handleChange('external_alert_api', 'url')}
              margin="normal"
              disabled={!licenseValid}
            />
            <TextField
              fullWidth
              label="Token"
              value={settings.external_alert_api.token}
              onChange={handleChange('external_alert_api', 'token')}
              margin="normal"
              disabled={!licenseValid}
            />
            <TextField
              fullWidth
              label="Secret"
              type="password"
              value={settings.external_alert_api.secret}
              onChange={handleChange('external_alert_api', 'secret')}
              margin="normal"
              disabled={!licenseValid}
            />
          </CardContent>
        </Card>
      </Grid>

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
              disabled={!licenseValid}
            />
            <TextField
              fullWidth
              type="number"
              label="图片质量(1-100)"
              value={settings.alert.image_quality}
              onChange={handleChange('alert', 'image_quality')}
              margin="normal"
              disabled={!licenseValid}
            />
          </CardContent>
        </Card>
      </Grid>

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
              disabled={!licenseValid}
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

      {isCustomerAdmin && (
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>服务运维</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                程序异常时可远程重启后端。平台与边缘均需 Docker Compose（restart: unless-stopped）部署才会自动拉起；
                边缘 Agent 请在「节点」页按台重启。
              </Typography>
              <Button
                variant="outlined"
                color="warning"
                startIcon={<RestartAlt />}
                onClick={handleRestartBackend}
                disabled={restartingBackend || !licenseValid}
              >
                {restartingBackend ? '正在重启…' : '重启管理平台后端'}
              </Button>
            </CardContent>
          </Card>
        </Grid>
      )}

      <Grid item xs={12}>
        <Button
          variant="contained"
          color="primary"
          startIcon={<Save />}
          onClick={handleSave}
          disabled={!licenseValid}
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
