import React, { useState, useEffect } from 'react';
import {
  Box, Card, CardContent, TextField, Button, Typography,
  Container, Alert, Avatar, Stack
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import axios, { getBaseUrl } from '../utils/axios';

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
    <Container maxWidth="sm">
      <Box sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <Card sx={{ width: '100%' }}>
          <CardContent>
            <Stack alignItems="center" spacing={1} sx={{ mb: 2 }}>
              {logoSrc && (
                <Avatar
                  src={logoSrc}
                  variant="rounded"
                  sx={{ width: 72, height: 72, bgcolor: 'transparent', mb: 1 }}
                />
              )}
              <Typography variant="h5" align="center">
                {branding.product_name}
              </Typography>
              {branding.company_name && (
                <Typography variant="body2" color="text.secondary" align="center">
                  {branding.company_name}
                </Typography>
              )}
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
                sx={{ mt: 2 }}
              >
                登录
              </Button>
            </form>
          </CardContent>
        </Card>
      </Box>
    </Container>
  );
}

export default Login;
