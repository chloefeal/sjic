import React, { useEffect, useState, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Box, Drawer, AppBar, Toolbar, List, Typography, ListItem, ListItemIcon, ListItemText, IconButton, Avatar, Stack
} from '@mui/material';
import {
  Videocam, ModelTraining, Settings, Build, NotificationsActive, Task, Code, Logout, Computer, Dashboard as DashboardIcon
} from '@mui/icons-material';
import axios, { getBaseUrl } from '../utils/axios';

const drawerWidth = 240;

const menuItems = [
  { text: '运行概览', icon: <DashboardIcon />, path: '/dashboard' },
  { text: '节点', icon: <Computer />, path: '/nodes' },
  { text: '视频源', icon: <Videocam />, path: '/streams' },
  { text: '模型管理', icon: <ModelTraining />, path: '/models', role: 'vendor' },
  { text: '算法清单', icon: <Code />, path: '/algorithms' },
  { text: '任务', icon: <Task />, path: '/tasks' },
  { text: '模型训练', icon: <Build />, path: '/training', role: 'vendor' },
  { text: '告警记录', icon: <NotificationsActive />, path: '/alerts' },
  { text: '系统设置', icon: <Settings />, path: '/settings' },
];

const DEFAULT_BRANDING = {
  company_name: '',
  product_name: '视觉检测系统',
  logo_url: '',
};

function Layout({ children }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [branding, setBranding] = useState(DEFAULT_BRANDING);

  const loadBranding = useCallback(async () => {
    try {
      const data = await axios.get('/api/branding');
      setBranding({
        company_name: data.company_name || '',
        product_name: data.product_name || DEFAULT_BRANDING.product_name,
        logo_url: data.logo_url || '',
      });
      if (data.product_name) {
        document.title = data.company_name
          ? `${data.product_name} · ${data.company_name}`
          : data.product_name;
      }
    } catch (e) {
      // 保持默认标题
    }
  }, []);

  useEffect(() => {
    loadBranding();
    const onUpdate = () => loadBranding();
    window.addEventListener('branding-updated', onUpdate);
    return () => window.removeEventListener('branding-updated', onUpdate);
  }, [loadBranding]);

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user_role');
    localStorage.removeItem('username');
    navigate('/login');
  };

  const userRole = localStorage.getItem('user_role');
  const filteredMenuItems = menuItems.filter(item => {
    if (item.role && item.role !== userRole) return false;
    return true;
  });

  const logoSrc = branding.logo_url ? `${getBaseUrl()}${branding.logo_url}` : '';

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar sx={{ justifyContent: 'space-between' }}>
          <Stack direction="row" spacing={1.5} alignItems="center">
            {logoSrc && (
              <Avatar
                src={logoSrc}
                variant="rounded"
                sx={{ width: 36, height: 36, bgcolor: 'transparent' }}
              />
            )}
            <Box>
              <Typography variant="h6" noWrap component="div" sx={{ lineHeight: 1.2 }}>
                {branding.product_name}
              </Typography>
              {branding.company_name && (
                <Typography variant="caption" sx={{ opacity: 0.75 }}>
                  {branding.company_name}
                </Typography>
              )}
            </Box>
          </Stack>
          <IconButton color="inherit" onClick={handleLogout}>
            <Logout />
          </IconButton>
        </Toolbar>
      </AppBar>
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
          },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto' }}>
          <List>
            {filteredMenuItems.map((item) => (
              <ListItem
                button
                key={item.text}
                selected={location.pathname === item.path}
                onClick={() => navigate(item.path)}
              >
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
}

export default Layout;
