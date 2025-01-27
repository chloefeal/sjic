import React, { useState, useEffect } from 'react';
import {
  Grid, Paper, Table, TableBody, TableCell, TableContainer, TableHead, 
  TableRow
} from '@mui/material';
import axios from '../utils/axios';

function Algorithms() {
  const [algorithms, setAlgorithms] = useState([]);

  useEffect(() => {
    fetchAlgorithms();
  }, []);

  const fetchAlgorithms = async () => {
    try {
      const response = await axios.get('/api/algorithms');
      setAlgorithms(response || []);
    } catch (error) {
      console.error('Error fetching algorithms:', error);
    }
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>ID</TableCell>
                <TableCell>名称</TableCell>
                <TableCell>类型</TableCell>
                <TableCell>描述</TableCell>
                <TableCell>参数配置</TableCell>
                <TableCell>创建时间</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {algorithms.map((algorithm) => (
                <TableRow key={algorithm.id}>
                  <TableCell>{algorithm.id}</TableCell>
                  <TableCell>{algorithm.name}</TableCell>
                  <TableCell>
                    {algorithm.type}
                  </TableCell>
                  <TableCell>{algorithm.description}</TableCell>
                  <TableCell>
                    <pre style={{ margin: 0 }}>
                      {JSON.stringify(algorithm.parameters, null, 2)}
                    </pre>
                  </TableCell>
                  <TableCell>
                    {new Date(algorithm.created_at).toLocaleString()}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Grid>
    </Grid>
  );
}

export default Algorithms; 