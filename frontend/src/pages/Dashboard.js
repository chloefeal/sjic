import React, { useEffect, useState, useRef } from 'react';
import {
  Box, Grid, Paper, Typography, Stack, Chip, Divider, Link as MuiLink
} from '@mui/material';
import {
  Videocam, ModelTraining, Code, Computer, Task, NotificationsActive
} from '@mui/icons-material';
import { Link as RouterLink } from 'react-router-dom';
import axios, { getBaseUrl } from '../utils/axios';

const REFRESH_MS = 15000;
const TICKER_MS = 4500;

const STAT_CARDS = [
  { key: 'cameras', label: '接入视频', icon: Videocam, path: '/streams', color: '#42a5f5' },
  { key: 'models', label: '模型数量', icon: ModelTraining, path: '/models', color: '#ab47bc' },
  { key: 'algorithms', label: '算法数量', icon: Code, path: '/algorithms', color: '#26a69a' },
  { key: 'nodes', label: '边缘终端', icon: Computer, path: '/nodes', color: '#66bb6a', subKey: 'nodes_online', subLabel: '在线' },
  { key: 'tasks_running', label: '运行中任务', icon: Task, path: '/tasks', color: '#ffa726', subKey: 'tasks', subLabel: '全部' },
];

function formatTime(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleString('zh-CN', { hour12: false });
  } catch {
    return iso;
  }
}

function Dashboard() {
  const [counts, setCounts] = useState({
    cameras: 0,
    models: 0,
    algorithms: 0,
    nodes: 0,
    nodes_online: 0,
    tasks: 0,
    tasks_running: 0,
    alerts: 0,
  });
  const [recentAlerts, setRecentAlerts] = useState([]);
  const [latestAlert, setLatestAlert] = useState(null);
  const [tickerIndex, setTickerIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const tickerRef = useRef(0);

  const fetchSummary = async () => {
    try {
      const data = await axios.get('/api/dashboard/summary');
      setCounts(data.counts || {});
      setRecentAlerts(data.recent_alerts || []);
      setLatestAlert(data.latest_alert || null);
    } catch (err) {
      console.error('Dashboard summary failed', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSummary();
    const timer = setInterval(fetchSummary, REFRESH_MS);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!recentAlerts.length) return undefined;
    const timer = setInterval(() => {
      tickerRef.current = (tickerRef.current + 1) % recentAlerts.length;
      setTickerIndex(tickerRef.current);
    }, TICKER_MS);
    return () => clearInterval(timer);
  }, [recentAlerts.length]);

  const tickerAlert = recentAlerts[tickerIndex] || null;
  const latestImage = latestAlert?.image_url
    ? `${getBaseUrl()}${latestAlert.image_url}`
    : null;

  return (
    <Box>
      <Stack direction="row" justifyContent="space-between" alignItems="baseline" sx={{ mb: 2 }}>
        <Typography variant="h5">运行概览</Typography>
        <Typography variant="caption" color="text.secondary">
          {loading ? '加载中…' : `每 ${REFRESH_MS / 1000}s 自动刷新`}
        </Typography>
      </Stack>

      <Box
        sx={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 2,
        }}
      >
        {STAT_CARDS.map((card) => {
          const Icon = card.icon;
          const value = counts[card.key] ?? 0;
          const sub = card.subKey != null ? counts[card.subKey] : null;
          return (
            <Paper
              key={card.key}
              component={RouterLink}
              to={card.path}
              elevation={0}
              sx={{
                p: 2,
                textDecoration: 'none',
                color: 'inherit',
                flex: '1 1 160px',
                minWidth: 160,
                border: '1px solid',
                borderColor: 'divider',
                background: 'linear-gradient(145deg, rgba(255,255,255,0.04) 0%, transparent 60%)',
                transition: 'border-color .2s, transform .2s',
                '&:hover': {
                  borderColor: card.color,
                  transform: 'translateY(-2px)',
                },
              }}
            >
              <Stack direction="row" spacing={1.5} alignItems="center">
                <Box
                  sx={{
                    width: 40,
                    height: 40,
                    borderRadius: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    bgcolor: `${card.color}22`,
                    color: card.color,
                  }}
                >
                  <Icon fontSize="small" />
                </Box>
                <Box>
                  <Typography variant="body2" color="text.secondary">{card.label}</Typography>
                  <Typography variant="h4" sx={{ lineHeight: 1.1, fontWeight: 600 }}>
                    {value}
                  </Typography>
                  {sub != null && (
                    <Typography variant="caption" color="text.secondary">
                      {card.subLabel} {sub}
                    </Typography>
                  )}
                </Box>
              </Stack>
            </Paper>
          );
        })}
      </Box>

      {/* 告警滚动条 */}
      <Paper
        elevation={0}
        sx={{
          mt: 2,
          px: 2,
          py: 1.25,
          border: '1px solid',
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          gap: 2,
          overflow: 'hidden',
          bgcolor: 'rgba(220, 0, 78, 0.08)',
        }}
      >
        <Chip
          size="small"
          color="secondary"
          icon={<NotificationsActive />}
          label={`告警 ${counts.alerts ?? 0}`}
          component={RouterLink}
          to="/alerts"
          clickable
        />
        <Box sx={{ flex: 1, minWidth: 0, overflow: 'hidden' }}>
          {tickerAlert ? (
            <Typography
              key={tickerAlert.id}
              variant="body2"
              noWrap
              sx={{
                animation: 'fadeSlide 0.45s ease',
                '@keyframes fadeSlide': {
                  from: { opacity: 0, transform: 'translateY(6px)' },
                  to: { opacity: 1, transform: 'translateY(0)' },
                },
              }}
            >
              <Box component="span" sx={{ color: 'text.secondary', mr: 1 }}>
                {formatTime(tickerAlert.timestamp)}
              </Box>
              [{tickerAlert.camera_name || `摄像头#${tickerAlert.camera_id}`}]{' '}
              {tickerAlert.message || tickerAlert.alert_type}
            </Typography>
          ) : (
            <Typography variant="body2" color="text.secondary">暂无告警</Typography>
          )}
        </Box>
        <MuiLink component={RouterLink} to="/alerts" underline="hover" variant="body2">
          全部记录
        </MuiLink>
      </Paper>

      <Grid container spacing={2} sx={{ mt: 0.5 }}>
        {/* 最新告警截图 */}
        <Grid item xs={12} md={7}>
          <Paper
            elevation={0}
            sx={{
              p: 2,
              border: '1px solid',
              borderColor: 'divider',
              height: '100%',
              minHeight: 320,
            }}
          >
            <Typography variant="h6" gutterBottom>最新告警</Typography>
            {latestAlert ? (
              <Stack spacing={1.5}>
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                  <Chip size="small" label={latestAlert.alert_type} color="secondary" />
                  <Chip
                    size="small"
                    variant="outlined"
                    label={latestAlert.camera_name || `摄像头#${latestAlert.camera_id}`}
                  />
                  <Typography variant="caption" color="text.secondary" sx={{ alignSelf: 'center' }}>
                    {formatTime(latestAlert.timestamp)}
                  </Typography>
                </Stack>
                <Typography variant="body2">
                  {latestAlert.message || '（无详细描述）'}
                </Typography>
                <Box
                  sx={{
                    mt: 1,
                    borderRadius: 1,
                    overflow: 'hidden',
                    bgcolor: 'grey.900',
                    border: '1px solid',
                    borderColor: 'divider',
                    minHeight: 220,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  {latestImage ? (
                    <Box
                      component="img"
                      src={latestImage}
                      alt="alert"
                      sx={{ width: '100%', maxHeight: 360, objectFit: 'contain', display: 'block' }}
                    />
                  ) : (
                    <Typography color="text.secondary">无截图</Typography>
                  )}
                </Box>
              </Stack>
            ) : (
              <Box sx={{ py: 8, textAlign: 'center' }}>
                <Typography color="text.secondary">当前没有告警记录</Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* 近期告警列表 */}
        <Grid item xs={12} md={5}>
          <Paper
            elevation={0}
            sx={{
              p: 2,
              border: '1px solid',
              borderColor: 'divider',
              height: '100%',
              minHeight: 320,
            }}
          >
            <Typography variant="h6" gutterBottom>近期告警</Typography>
            <Divider sx={{ mb: 1 }} />
            <Stack spacing={0} sx={{ maxHeight: 400, overflow: 'auto' }}>
              {recentAlerts.length === 0 && (
                <Typography variant="body2" color="text.secondary" sx={{ py: 2 }}>
                  暂无数据
                </Typography>
              )}
              {recentAlerts.slice(0, 12).map((alert, idx) => (
                <Box key={alert.id}>
                  {idx > 0 && <Divider />}
                  <Box sx={{ py: 1.25 }}>
                    <Stack direction="row" justifyContent="space-between" spacing={1}>
                      <Typography variant="body2" noWrap sx={{ fontWeight: 500, flex: 1 }}>
                        {alert.message || alert.alert_type}
                      </Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ whiteSpace: 'nowrap' }}>
                        {formatTime(alert.timestamp)}
                      </Typography>
                    </Stack>
                    <Typography variant="caption" color="text.secondary">
                      {alert.camera_name || `摄像头#${alert.camera_id}`} · {alert.alert_type}
                    </Typography>
                  </Box>
                </Box>
              ))}
            </Stack>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Dashboard;
