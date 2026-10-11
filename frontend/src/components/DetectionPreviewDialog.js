import React, { useEffect, useState } from 'react';
import {
  Alert, Box, Button, Chip, Dialog, DialogActions, DialogContent, DialogTitle,
  Typography
} from '@mui/material';
import axios from '../utils/axios';

export function taskPreviewEngines(task) {
  if (!task) return [];
  if (task.pipeline === 'exam' || task.algorithm_engine === 'exam_pipeline' || task.od_algorithm_id || task.pose_algorithm_id) {
    const engines = [];
    if (task.od_algorithm_id) engines.push('object_detection');
    if (task.pose_algorithm_id) engines.push('pose_behavior');
    if (engines.length) return engines;
  }
  if (task.algorithm_engine === 'object_detection' || task.algorithm_engine === 'pose_behavior') {
    return [task.algorithm_engine];
  }
  return [];
}

const STATUS_LABEL = {
  running: '运行中',
  stopped: '已停止',
  syncing: '同步中',
  starting: '启动中',
  error: '出错',
};

function formatScore(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  return number.toFixed(2);
}

function extractApiError(error, fallback) {
  const data = error?.response?.data;
  if (typeof data?.error === 'string' && data.error) return data.error;
  return fallback;
}

function CountBlock({ title, data, emptyText, noun, hint }) {
  if (!data) {
    return (
      <Box sx={{ flex: 1, minWidth: 220 }}>
        <Typography variant="subtitle1">{title}</Typography>
        <Typography variant="body2" color="text.secondary">这一帧还没有{title}结果</Typography>
      </Box>
    );
  }
  if (data.ok === false) {
    return (
      <Box sx={{ flex: 1, minWidth: 220 }}>
        <Typography variant="subtitle1">{title}</Typography>
        <Typography variant="body2" color="error">这一帧推理失败</Typography>
      </Box>
    );
  }
  const classes = Array.isArray(data.classes) ? data.classes : [];
  return (
    <Box sx={{ flex: 1, minWidth: 220 }}>
      <Typography variant="subtitle1">{title}</Typography>
      <Typography variant="h5" sx={{ my: 0.5 }}>
        {data.count ?? 0}
        <Typography component="span" variant="body2" color="text.secondary" sx={{ ml: 1 }}>
          个超过阈值的{noun}
        </Typography>
      </Typography>
      <Typography variant="body2" color="text.secondary">
        推理阈值 {formatScore(data.threshold)}
        {data.with_skeleton != null ? ` · 有骨骼点 ${data.with_skeleton}` : ''}
        {data.max_confidence ? ` · 最高 ${formatScore(data.max_confidence)}` : ''}
      </Typography>
      {hint && (
        <Typography variant="body2" color="text.secondary">{hint}</Typography>
      )}
      {data.shown != null && data.count > data.shown && (
        <Typography variant="body2" color="text.secondary">
          画面上画出置信度最高的 {data.shown} 个，其余只计入上面的数量
        </Typography>
      )}
      {classes.length > 0 && (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mt: 1 }}>
          {classes.map((item) => (
            <Chip
              key={`${item.id}-${item.name}`}
              size="small"
              variant="outlined"
              label={`${item.name} ${item.count} · 最高 ${formatScore(item.max_confidence)}`}
            />
          ))}
        </Box>
      )}
      {!classes.length && (data.count ?? 0) === 0 && (
        <Typography variant="body2" sx={{ mt: 1 }}>{emptyText}</Typography>
      )}
    </Box>
  );
}

function DetectionPreviewDialog({ task, onClose }) {
  const [image, setImage] = useState(null);
  const [stats, setStats] = useState(null);
  const [ageMs, setAgeMs] = useState(null);
  const [waiting, setWaiting] = useState(true);
  const [armError, setArmError] = useState('');
  const [pollError, setPollError] = useState('');
  const [session, setSession] = useState(null);
  const [waitedLong, setWaitedLong] = useState(false);
  const engines = session?.engines?.length ? session.engines : taskPreviewEngines(task);

  useEffect(() => {
    if (!task?.id) return undefined;
    let stopped = false;
    const view = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    setImage(null);
    setStats(null);
    setAgeMs(null);
    setWaiting(true);
    setArmError('');
    setPollError('');
    setSession(null);
    setWaitedLong(false);

    const arm = async () => {
      if (stopped) return;
      const issuedAt = Date.now();
      try {
        const data = await axios.post(`/api/tasks/${task.id}/preview`, {
          enabled: true,
          view,
          issued_at: issuedAt,
        });
        if (!stopped) {
          setSession(data || null);
          setArmError('');
        }
      } catch (error) {
        if (!stopped) setArmError(extractApiError(error, '打不开检测预览'));
      }
    };

    const poll = async () => {
      try {
        const data = await axios.get(`/api/tasks/${task.id}/preview`);
        if (stopped) return;
        setWaiting(Boolean(data?.waiting) || !data?.image);
        setStats(data?.stats || null);
        setAgeMs(Number.isFinite(data?.age_ms) ? data.age_ms : null);
        setImage(data?.image || null);
        setPollError('');
      } catch (error) {
        if (!stopped) setPollError(extractApiError(error, '刷新检测画面失败'));
      }
    };

    arm();
    poll();
    const keepTimer = setInterval(arm, 3000);
    const pollTimer = setInterval(poll, 400);
    const waitTimer = setTimeout(() => setWaitedLong(true), 8000);
    return () => {
      stopped = true;
      clearInterval(keepTimer);
      clearInterval(pollTimer);
      clearTimeout(waitTimer);
      axios.post(`/api/tasks/${task.id}/preview`, {
        enabled: false,
        view,
        issued_at: Date.now(),
      }).catch(() => {});
    };
  }, [task?.id]);

  if (!task) return null;

  const inferMs = stats?.infer_ms;
  const stale = !waiting && ageMs != null && ageMs > 4000;
  const taskRunning = task.status === 'running';

  return (
    <Dialog open onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>检测预览 — {task.name}</DialogTitle>
      <DialogContent>
        <Alert severity="info" sx={{ mb: 1.5 }}>
          画面是边缘正在推理的那一帧。目标检测画框并标出类别，姿态检测画骨骼点。
          下面的数量是达到推理阈值的候选，场景告警还会再按区域和停留时间过滤，所以这里有框不一定已经告警。
        </Alert>
        {armError && <Alert severity="error" sx={{ mb: 1.5 }}>{armError}</Alert>}
        {!armError && session && session.node_status && session.node_status !== 'online' && (
          <Alert severity="warning" sx={{ mb: 1.5 }}>算力节点不在线，预览指令到不了边缘。</Alert>
        )}
        {!armError && !taskRunning && (
          <Alert severity="warning" sx={{ mb: 1.5 }}>
            任务当前是「{STATUS_LABEL[task.status] || task.status || '未运行'}」。先启动任务，节点开始推理后这里才会有画面。
          </Alert>
        )}
        <Box
          sx={{
            bgcolor: '#000',
            minHeight: 280,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mb: 2,
          }}
        >
          {image ? (
            <img
              src={`data:image/jpeg;base64,${image}`}
              alt="实时检测画面"
              style={{ width: '100%', maxHeight: '68vh', objectFit: 'contain', display: 'block' }}
            />
          ) : (
            <Box sx={{ px: 2, py: 6, textAlign: 'center' }}>
              <Typography color="grey.400">正在等待边缘回传检测画面…</Typography>
              {waitedLong && (
                <Typography color="grey.500" variant="body2" sx={{ mt: 1 }}>
                  确认任务已启动，并且算力节点在线。节点离线或任务没在推理时，这里不会有画面。
                </Typography>
              )}
            </Box>
          )}
        </Box>
        {pollError && <Alert severity="warning" sx={{ mb: 1.5 }}>{pollError}</Alert>}
        {stale && (
          <Alert severity="warning" sx={{ mb: 1.5 }}>
            已经 {(ageMs / 1000).toFixed(1)} 秒没有新画面。任务可能停了，或节点暂时没推上来。
          </Alert>
        )}
        {image && (
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
            {engines.includes('object_detection') && (
              <CountBlock
                title="目标检测"
                noun="目标"
                data={stats?.object}
                emptyText="这一帧没有超过阈值的目标"
                hint="只包含这个任务选过的类别"
              />
            )}
            {engines.includes('pose_behavior') && (
              <CountBlock
                title="姿态检测"
                noun="人体"
                data={stats?.pose}
                emptyText="这一帧没有超过阈值的人体"
              />
            )}
          </Box>
        )}
        {(inferMs != null || ageMs != null) && (
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1.5 }}>
            {inferMs != null ? `推理 ${inferMs} ms` : ''}
            {inferMs != null && ageMs != null ? ' · ' : ''}
            {ageMs != null ? `画面 ${(ageMs / 1000).toFixed(1)} 秒前` : ''}
          </Typography>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>关闭</Button>
      </DialogActions>
    </Dialog>
  );
}

export default DetectionPreviewDialog;
