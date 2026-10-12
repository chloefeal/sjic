import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert, Box, Button, Card, CardContent, Chip, IconButton, Stack, Typography,
} from '@mui/material';
import { CloudUpload, Delete } from '@mui/icons-material';
import axios from '../utils/axios';
import ScrollableDataPage from '../components/ScrollableDataPage';

const SET_ID = 'default';

function ImageTest() {
  const fileRef = useRef(null);
  const imagesRef = useRef([]);
  const [images, setImages] = useState([]);
  const [current, setCurrent] = useState(null);
  const [camera, setCamera] = useState(null);
  const [cameraError, setCameraError] = useState('');
  const [thumbs, setThumbs] = useState({});
  const [error, setError] = useState('');
  const [uploading, setUploading] = useState(false);
  const [busyName, setBusyName] = useState('');

  const namesKey = useMemo(() => images.map((item) => item.name).join('|'), [images]);
  imagesRef.current = images;

  const apply = (data, keepCameraError = false) => {
    setImages(data?.images || []);
    setCurrent(data?.current || null);
    if (data && Object.prototype.hasOwnProperty.call(data, 'camera')) {
      setCamera(data.camera || null);
    }
    if (data && Object.prototype.hasOwnProperty.call(data, 'camera_error')) {
      setCameraError(data.camera_error || '');
    } else if (!keepCameraError) {
      setCameraError('');
    }
    if (Array.isArray(data?.errors) && data.errors.length > 0) {
      const text = data.errors.map((item) => `${item.filename}: ${item.error}`).join('；');
      setError(text);
    }
  };

  const load = async () => {
    try {
      const data = await axios.get(`/api/image-sets/${SET_ID}`);
      apply(data, true);
      setError('');
    } catch (err) {
      setError(err.response?.data?.error || '加载图片失败');
    }
  };

  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    let cancelled = false;
    const created = [];
    const list = imagesRef.current;
    if (!list.length) {
      setThumbs({});
      return undefined;
    }
    Promise.all(list.map(async (image) => {
      try {
        const blob = await axios.get(
          `/api/image-sets/${SET_ID}/images/${image.name}`,
          { responseType: 'blob' },
        );
        const url = URL.createObjectURL(blob);
        created.push(url);
        return [image.name, url];
      } catch {
        return null;
      }
    })).then((pairs) => {
      if (cancelled) {
        created.forEach((url) => URL.revokeObjectURL(url));
        return;
      }
      const ready = pairs.filter(Boolean);
      if (ready.length !== list.length) setError('部分缩略图加载失败');
      setThumbs(Object.fromEntries(ready));
    });
    return () => {
      cancelled = true;
      created.forEach((url) => URL.revokeObjectURL(url));
    };
  }, [namesKey]);

  const upload = async (event) => {
    const files = Array.from(event.target.files || []);
    event.target.value = '';
    if (!files.length) return;
    const form = new FormData();
    files.forEach((file) => form.append('files', file));
    setUploading(true);
    setError('');
    try {
      const data = await axios.post(`/api/image-sets/${SET_ID}/images`, form, { timeout: 120000 });
      apply(data, true);
    } catch (err) {
      setError(err.response?.data?.error || '上传失败');
    } finally {
      setUploading(false);
    }
  };

  const setLabel = async (image, label) => {
    setBusyName(image.name);
    setError('');
    try {
      const data = await axios.put(`/api/image-sets/${SET_ID}/images/${image.name}`, { label });
      apply(data, true);
    } catch (err) {
      setError(err.response?.data?.error || '标记失败');
    } finally {
      setBusyName('');
    }
  };

  const selectImage = async (image) => {
    setBusyName(image.name);
    setError('');
    try {
      const data = await axios.post(`/api/image-sets/${SET_ID}/current`, { name: image.name });
      apply(data);
    } catch (err) {
      setError(err.response?.data?.error || '送入失败');
    } finally {
      setBusyName('');
    }
  };

  const removeImage = async (image) => {
    if (!window.confirm(`删除「${image.original || image.name}」？`)) return;
    setBusyName(image.name);
    setError('');
    try {
      const data = await axios.delete(`/api/image-sets/${SET_ID}/images/${image.name}`);
      apply(data, true);
    } catch (err) {
      setError(err.response?.data?.error || '删除失败');
    } finally {
      setBusyName('');
    }
  };

  const cameraHint = camera
    ? `视频源「${camera.name}」已就绪，地址 ${camera.url}，机位 ${camera.mount_position_label || '后上方'}。若节点已经绑定过视频源，请到「节点」把这一路也绑上，任务里才能选到。同一时刻只有这一路假流。换图不会清空跟踪和计时，要单独重测请先停止任务再启动。`
    : '点「送入算法」后会创建视频源「图片测试」，机位默认后上方，可在视频源页面修改。启动任务前需要先送入一张图。';

  return (
    <ScrollableDataPage
      header={(
        <Box sx={{ mb: 2 }}>
          <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={2}>
            <Typography variant="h5">图片测试</Typography>
            <Button
              variant="contained"
              startIcon={<CloudUpload />}
              disabled={uploading}
              onClick={() => fileRef.current?.click()}
            >
              {uploading ? '上传中…' : '上传图片'}
            </Button>
            <input
              ref={fileRef}
              type="file"
              hidden
              multiple
              accept=".png,.jpg,.jpeg,.bmp,.webp,image/png,image/jpeg,image/bmp,image/webp"
              onChange={upload}
            />
          </Stack>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            用正常和异常样本冒充一路视频。算法、告警和任务都走原来的流程，不会为每张图开一路流。
          </Typography>
          <Alert severity="info" sx={{ mt: 2 }}>{cameraHint}</Alert>
          {cameraError && <Alert severity="warning" sx={{ mt: 1 }}>{cameraError}</Alert>}
          {error && <Alert severity="error" sx={{ mt: 1 }}>{error}</Alert>}
        </Box>
      )}
    >
      <Box sx={{ overflow: 'auto', flex: 1, minHeight: 0, pb: 2 }}>
        {images.length === 0 && (
          <Typography variant="body2" color="text.secondary">
            还没有图片。上传后点「送入算法」，再在任务里选择视频源「图片测试」。
          </Typography>
        )}
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
            gap: 2,
          }}
        >
          {images.map((image) => {
            const active = image.name === current;
            const busy = busyName === image.name;
            return (
              <Card
                key={image.name}
                variant="outlined"
                sx={{
                  borderColor: active ? 'primary.main' : 'divider',
                  borderWidth: active ? 2 : 1,
                }}
              >
                <Box
                  sx={{
                    height: 150,
                    bgcolor: '#111',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  {thumbs[image.name] ? (
                    <Box
                      component="img"
                      src={thumbs[image.name]}
                      alt={image.original || image.name}
                      sx={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                    />
                  ) : (
                    <Typography variant="caption" color="grey.500">加载中</Typography>
                  )}
                </Box>
                <CardContent sx={{ pb: 1.5 }}>
                  <Stack direction="row" alignItems="flex-start" spacing={1}>
                    <Typography variant="body2" sx={{ flex: 1, wordBreak: 'break-all' }}>
                      {image.original || image.name}
                    </Typography>
                    <IconButton size="small" disabled={busy} onClick={() => removeImage(image)}>
                      <Delete fontSize="small" />
                    </IconButton>
                  </Stack>
                  <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                    <Chip
                      size="small"
                      label="正常"
                      color={image.label === 'normal' ? 'success' : 'default'}
                      variant={image.label === 'normal' ? 'filled' : 'outlined'}
                      onClick={() => setLabel(image, image.label === 'normal' ? '' : 'normal')}
                      disabled={busy}
                    />
                    <Chip
                      size="small"
                      label="异常"
                      color={image.label === 'abnormal' ? 'warning' : 'default'}
                      variant={image.label === 'abnormal' ? 'filled' : 'outlined'}
                      onClick={() => setLabel(image, image.label === 'abnormal' ? '' : 'abnormal')}
                      disabled={busy}
                    />
                    {active && <Chip size="small" color="primary" label="当前帧" />}
                  </Stack>
                  <Button
                    fullWidth
                    sx={{ mt: 1.5 }}
                    variant={active ? 'outlined' : 'contained'}
                    disabled={busy || active}
                    onClick={() => selectImage(image)}
                  >
                    {active ? '正在送入' : '送入算法'}
                  </Button>
                </CardContent>
              </Card>
            );
          })}
        </Box>
      </Box>
    </ScrollableDataPage>
  );
}

export default ImageTest;
