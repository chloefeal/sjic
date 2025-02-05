import React, { useState, useEffect } from 'react';
import { Box, TextField, Button } from '@mui/material';

function CalibrationTool({ onCalibrate }) {
  const [frame, setFrame] = useState(null);
  const [points, setPoints] = useState([]);
  const [beltWidth, setBeltWidth] = useState('');

  // 获取相机预览帧
  useEffect(() => {
    // ... 获取相机预览帧的代码
  }, []);

  const handleImageClick = (event) => {
    const rect = event.target.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    if (points.length < 2) {
      setPoints([...points, { x, y }]);
    }
  };

  const handleCalibrate = () => {
    if (points.length === 2 && beltWidth) {
      const pixelDistance = Math.sqrt(
        Math.pow(points[1].x - points[0].x, 2) +
        Math.pow(points[1].y - points[0].y, 2)
      );
      const pixelToCm = parseFloat(beltWidth) / pixelDistance;
      
      onCalibrate({
        pixel_to_cm: pixelToCm,
        calibration: {
          belt_width: parseFloat(beltWidth),
          points: points
        }
      });
    }
  };

  return (
    <Box>
      <Box
        component="img"
        src={frame}
        onClick={handleImageClick}
        sx={{ cursor: 'crosshair' }}
      />
      {points.map((point, index) => (
        <Box
          key={index}
          sx={{
            position: 'absolute',
            left: point.x - 5,
            top: point.y - 5,
            width: 10,
            height: 10,
            borderRadius: '50%',
            bgcolor: 'red',
          }}
        />
      ))}
      <TextField
        label="皮带实际宽度(cm)"
        type="number"
        value={beltWidth}
        onChange={(e) => setBeltWidth(e.target.value)}
      />
      <Button
        onClick={handleCalibrate}
        disabled={points.length !== 2 || !beltWidth}
      >
        计算转换比例
      </Button>
    </Box>
  );
}

export default CalibrationTool; 