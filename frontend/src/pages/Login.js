import React, { useState, useEffect } from 'react';
import {
  Box, Card, CardContent, TextField, Button, Typography,
  Alert, Avatar, Stack, keyframes
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import { useNavigate } from 'react-router-dom';
import axios, { getBaseUrl } from '../utils/axios';

const pulse = keyframes`
  0%, 100% { opacity: 0.45; transform: scale(1); }
  50% { opacity: 0.75; transform: scale(1.05); }
`;

const scan = keyframes`
  0% { transform: translateY(-100%); }
  100% { transform: translateY(100vh); }
`;

function Login() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });
  const [error, setError] = useState('');
  const [branding, setBranding] = useState({
    company_name: '',
    product_name: '智算检测平台',
    logo_url: '',
  });

  useEffect(() => {
    axios.get('/api/branding')
      .then((data) => {
        setBranding({
          company_name: data.company_name || '',
          product_name: data.product_name || '智算检测平台',
          logo_url: data.logo_url || '',
        });
        if (data.product_name) {
          document.title = data.company_name
            ? `${data.product_name} · ${data.company_name}`
            : data.product_name;
        }
      })
      .catch(() => {});
  }, []);

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('/api/login', formData);
      localStorage.setItem('token', response.token);
      localStorage.setItem('user_role', response.role);
      localStorage.setItem('username', response.username);
      navigate('/dashboard');
    } catch (err) {
      setError('用户名或密码错误');
    }
  };

  const logoSrc = branding.logo_url ? `${getBaseUrl()}${branding.logo_url}` : '';

  return (
    <Box
      sx={{
        minHeight: '100vh',
        position: 'relative',
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        px: 2,
        backgroundColor: '#050912',
        backgroundImage: `
          radial-gradient(ellipse 70% 55% at 15% 20%, ${alpha('#00C8F0', 0.22)}, transparent 55%),
          radial-gradient(ellipse 50% 45% at 85% 75%, ${alpha('#FFB020', 0.14)}, transparent 50%),
          radial-gradient(ellipse 40% 35% at 70% 15%, ${alpha('#00C8F0', 0.1)}, transparent 45%),
          linear-gradient(${alpha('#00C8F0', 0.05)} 1px, transparent 1px),
          linear-gradient(90deg, ${alpha('#00C8F0', 0.05)} 1px, transparent 1px)
        `,
        backgroundSize: 'auto, auto, auto, 56px 56px, 56px 56px',
      }}
    >
      {/* Ambient orbs */}
      <Box
        sx={{
          position: 'absolute',
          width: 420,
          height: 420,
          borderRadius: '50%',
          top: '8%',
          left: '12%',
          background: `radial-gradient(circle, ${alpha('#00C8F0', 0.25)} 0%, transparent 70%)`,
          filter: 'blur(8px)',
          animation: `${pulse} 8s ease-in-out infinite`,
          pointerEvents: 'none',
        }}
      />
      <Box
        sx={{
          position: 'absolute',
          width: 320,
          height: 320,
          borderRadius: '50%',
          bottom: '10%',
          right: '10%',
          background: `radial-gradient(circle, ${alpha('#FFB020', 0.18)} 0%, transparent 70%)`,
          filter: 'blur(10px)',
          animation: `${pulse} 10s ease-in-out infinite reverse`,
          pointerEvents: 'none',
        }}
      />

      {/* Scan line */}
      <Box
        sx={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          overflow: 'hidden',
          '&::after': {
            content: '""',
            position: 'absolute',
            left: 0,
            right: 0,
            height: 120,
            background: `linear-gradient(180deg, transparent, ${alpha('#00C8F0', 0.08)}, transparent)`,
            animation: `${scan} 9s linear infinite`,
          },
        }}
      />

      <Card
        sx={{
          width: '100%',
          maxWidth: 420,
          position: 'relative',
          zIndex: 1,
          backgroundColor: alpha('#0A1220', 0.72),
          backdropFilter: 'blur(18px)',
          border: `1px solid ${alpha('#00C8F0', 0.28)}`,
          boxShadow: `
            0 0 0 1px ${alpha('#FFB020', 0.08)},
            0 24px 64px ${alpha('#000000', 0.55)},
            0 0 40px ${alpha('#00C8F0', 0.12)}
          `,
        }}
      >
        <CardContent sx={{ p: { xs: 3, sm: 4 } }}>
          <Stack alignItems="center" spacing={1} sx={{ mb: 3 }}>
            {logoSrc && (
              <Avatar
                src={logoSrc}
                variant="rounded"
                sx={{ width: 72, height: 72, bgcolor: 'transparent', mb: 0.5 }}
              />
            )}
            <Typography
              variant="h5"
              align="center"
              sx={{
                background: 'linear-gradient(90deg, #E8F1FF 0%, #00C8F0 55%, #FFB020 100%)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                color: 'transparent',
              }}
            >
              {branding.product_name}
            </Typography>
            {branding.company_name && (
              <Typography variant="body2" color="text.secondary" align="center">
                {branding.company_name}
              </Typography>
            )}
            <Typography
              variant="caption"
              sx={{ color: alpha('#00C8F0', 0.75), letterSpacing: '0.18em', mt: 0.5 }}
            >
              INTELLIGENT VISION PLATFORM
            </Typography>
          </Stack>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}
          <form onSubmit={handleLogin}>
            <TextField
              fullWidth
              label="用户名"
              margin="normal"
              value={formData.username}
              onChange={(e) => setFormData({
                ...formData,
                username: e.target.value
              })}
            />
            <TextField
              fullWidth
              label="密码"
              type="password"
              margin="normal"
              value={formData.password}
              onChange={(e) => setFormData({
                ...formData,
                password: e.target.value
              })}
            />
            <Button
              fullWidth
              variant="contained"
              type="submit"
              size="large"
              sx={{ mt: 3, py: 1.2 }}
            >
              登录
            </Button>
          </form>
        </CardContent>
      </Card>
    </Box>
  );
}

export default Login;
