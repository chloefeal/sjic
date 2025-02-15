import React, { useState, useRef, useEffect } from 'react';
import { 
  Button, Dialog, DialogTitle, DialogContent, DialogActions,
  TextField, Paper
} from '@mui/material';
import axios from '../utils/axios';

function BeltCalibrationTool({ cameraId, onCalibrate }) {
  const [open, setOpen] = useState(false);
  const [imageUrl, setImageUrl] = useState(null);
  const [points, setPoints] = useState([]);
  const [beltWidth, setBeltWidth] = useState(0); // 真实宽度(cm)
  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  // 获取视频流截图
  const captureFrame = async () => {
    try {
      const response = await axios.post('/api/cameras/capture', {
        camera_id: cameraId
      }, {
        responseType: 'blob'
      });
      setImageUrl(URL.createObjectURL(response.data));
    } catch (error) {
      console.error('Error capturing frame:', error);
    }
  };

  // 处理画布点击
  const handleCanvasClick = (event) => {
    if (points.length >= 2) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    setPoints([...points, { x, y }]);
  };

  // 绘制画布
  const drawCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const image = imageRef.current;

    // 清空画布
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 绘制图片
    if (image) {
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    }

    // 绘制点和线
    points.forEach(point => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
      ctx.fillStyle = 'red';
      ctx.fill();
    });

    if (points.length === 2) {
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      ctx.lineTo(points[1].x, points[1].y);
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  };

  // 计算标定结果
  const handleCalibrate = () => {
    if (points.length !== 2 || !beltWidth) return;

    // 计算像素距离
    const pixelWidth = Math.sqrt(
      Math.pow(points[1].x - points[0].x, 2) + 
      Math.pow(points[1].y - points[0].y, 2)
    );

    // 计算像素到厘米的转换比例
    const pixel_to_cm = beltWidth / pixelWidth;

    onCalibrate({
      pixel_to_cm,
      calibration: {
        belt_width: beltWidth,
        points: points
      }
    });
    setOpen(false);
  };

  // 重置标定
  const handleReset = () => {
    setPoints([]);
    setBeltWidth(0);
  };

  useEffect(() => {
    if (imageUrl) {
      const image = new Image();
      image.src = imageUrl;
      image.onload = () => {
        imageRef.current = image;
        drawCanvas();
      };
    }
  }, [imageUrl]);

  useEffect(() => {
    drawCanvas();
  }, [points]);

  return (
    <>
      <Button variant="outlined" onClick={() => setOpen(true)}>
        皮带标定
      </Button>

      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>皮带宽度标定</DialogTitle>
        <DialogContent>
          <Paper style={{ padding: 16, marginBottom: 16 }}>
            <Button 
              variant="contained" 
              onClick={captureFrame}
              disabled={!cameraId}
              style={{ marginRight: 16 }}
            >
              截取当前帧
            </Button>
            <Button 
              variant="outlined" 
              onClick={handleReset}
              disabled={!imageUrl}
            >
              重置
            </Button>
          </Paper>

          {imageUrl && (
            <>
              <canvas
                ref={canvasRef}
                width={800}
                height={600}
                onClick={handleCanvasClick}
                style={{ border: '1px solid #ccc', marginBottom: 16 }}
              />
              <TextField
                fullWidth
                type="number"
                label="皮带实际宽度(cm)"
                value={beltWidth}
                onChange={(e) => setBeltWidth(parseFloat(e.target.value))}
                inputProps={{ min: 0, step: 0.1 }}
              />
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)}>取消</Button>
          <Button 
            onClick={handleCalibrate}
            variant="contained"
            disabled={points.length !== 2 || !beltWidth}
          >
            确定
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

export default BeltCalibrationTool; 