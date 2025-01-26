import React, { useState, useEffect } from 'react';
import { Grid, Card, CardContent, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from '@mui/material';
import io from 'socket.io-client';
import axios from '../utils/axios';

function Alerts() {
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    fetchAlerts();
    // 设置WebSocket监听实时告警
    const socket = io(process.env.REACT_APP_API_URL);
    socket.on('new_alert', (alert) => {
      setAlerts((prev) => [alert, ...prev]);
    });

    return () => socket.disconnect();
  }, []);

  const fetchAlerts = async () => {
    try {
      const response = await axios.get('/api/alerts');
      setAlerts(response || []);  // 确保始终是数组
      console.log('Alerts:', response);
    } catch (error) {
      console.error('Error fetching alerts:', error);
    }
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              告警记录
            </Typography>
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>时间</TableCell>
                    <TableCell>摄像头</TableCell>
                    <TableCell>类型</TableCell>
                    <TableCell>置信度</TableCell>
                    <TableCell>图片</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {alerts.map((alert) => (
                    <TableRow key={alert.id}>
                      <TableCell>{new Date(alert.timestamp).toLocaleString()}</TableCell>
                      <TableCell>{alert.camera_name}</TableCell>
                      <TableCell>{alert.alert_type}</TableCell>
                      <TableCell>{(alert.confidence * 100).toFixed(2)}%</TableCell>
                      <TableCell>
                        <img
                          src={alert.image_url}
                          alt="告警截图"
                          style={{ width: 100, height: 'auto', cursor: 'pointer' }}
                          onClick={() => {/* 显示大图 */}}
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
}

export default Alerts; 