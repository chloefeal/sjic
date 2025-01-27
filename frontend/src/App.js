import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Layout from './components/Layout';
import VideoStreams from './pages/VideoStreams';
import Models from './pages/Models';
import Algorithms from './pages/Algorithms';
import Training from './pages/Training';
import Alerts from './pages/Alerts';

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
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Navigate to="/streams" />} />
            <Route path="/streams" element={<VideoStreams />} />
            <Route path="/models" element={<Models />} />
            <Route path="/algorithms" element={<Algorithms />} />
            <Route path="/training" element={<Training />} />
            <Route path="/alerts" element={<Alerts />} />
          </Routes>
        </Layout>
      </Router>
    </ThemeProvider>
  );
}

export default App; 