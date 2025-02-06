import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Layout from './components/Layout';
import PrivateRoute from './components/PrivateRoute';
import Login from './pages/Login';
import VideoStreams from './pages/VideoStreams';
import Models from './pages/Models';
import Tasks from './pages/Tasks';
import Training from './pages/Training';
import Alerts from './pages/Alerts';
import Algorithms from './pages/Algorithms';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={
            <PrivateRoute>
              <Layout>
                <Navigate to="/streams" />
              </Layout>
            </PrivateRoute>
          } />
          <Route path="/streams" element={
            <PrivateRoute>
              <Layout>
                <VideoStreams />
              </Layout>
            </PrivateRoute>
          } />
          <Route path="/streams" element={<VideoStreams />} />
          <Route path="/models" element={<Models />} />
          <Route path="/tasks" element={<Tasks />} />
          <Route path="/training" element={<Training />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/algorithms" element={<Algorithms />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App; 