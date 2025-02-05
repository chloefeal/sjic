import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Box, Drawer, AppBar, Toolbar, List, Typography, Divider, ListItem, ListItemIcon, ListItemText, IconButton } from '@mui/material';
import { Videocam, ModelTraining, Settings, Build, NotificationsActive, Task, Code, Logout } from '@mui/icons-material';

const drawerWidth = 240;

const menuItems = [
  { text: '视频流', icon: <Videocam />, path: '/streams' },
  { text: '模型管理', icon: <ModelTraining />, path: '/models' },
  { text: '算法管理', icon: <Code />, path: '/algorithms' },
  { text: '任务', icon: <Task />, path: '/tasks' },
  { text: '模型训练', icon: <Build />, path: '/training' },
  { text: '告警记录', icon: <NotificationsActive />, path: '/alerts' },
];

function Layout({ children }) {
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/login');
  };

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar sx={{ justifyContent: 'space-between' }}>
          <Typography variant="h6" noWrap component="div">
            视觉检测系统
          </Typography>
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
            {menuItems.map((item) => (
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