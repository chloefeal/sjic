import { createTheme, alpha } from '@mui/material/styles';

/** 科技风深色主题：青蓝主色 + 琥珀辅色 */
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00C8F0',
      light: '#5CDEFF',
      dark: '#0090B0',
      contrastText: '#031018',
    },
    secondary: {
      main: '#FFB020',
      light: '#FFC95C',
      dark: '#CC8800',
      contrastText: '#1A1200',
    },
    background: {
      default: '#070B14',
      paper: '#0E1524',
    },
    text: {
      primary: '#E8F1FF',
      secondary: '#8BA0BC',
    },
    divider: alpha('#00C8F0', 0.12),
    success: { main: '#22C55E' },
    warning: { main: '#FFB020' },
    error: { main: '#FF5C7A' },
    info: { main: '#38BDF8' },
  },
  typography: {
    fontFamily: '"Noto Sans SC", "Segoe UI", sans-serif',
    h5: {
      fontFamily: '"Orbitron", "Noto Sans SC", sans-serif',
      fontWeight: 600,
      letterSpacing: '0.04em',
    },
    h6: {
      fontFamily: '"Orbitron", "Noto Sans SC", sans-serif',
      fontWeight: 600,
      letterSpacing: '0.03em',
    },
    button: {
      fontWeight: 600,
      letterSpacing: '0.06em',
    },
  },
  shape: {
    borderRadius: 10,
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundImage: `
            radial-gradient(ellipse 80% 50% at 10% -10%, ${alpha('#00C8F0', 0.12)}, transparent 55%),
            radial-gradient(ellipse 60% 40% at 100% 0%, ${alpha('#FFB020', 0.06)}, transparent 50%),
            linear-gradient(${alpha('#00C8F0', 0.03)} 1px, transparent 1px),
            linear-gradient(90deg, ${alpha('#00C8F0', 0.03)} 1px, transparent 1px)
          `,
          backgroundSize: 'auto, auto, 48px 48px, 48px 48px',
          backgroundAttachment: 'fixed',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: alpha('#0A101C', 0.85),
          backdropFilter: 'blur(12px)',
          borderBottom: `1px solid ${alpha('#00C8F0', 0.18)}`,
          boxShadow: `0 0 24px ${alpha('#00C8F0', 0.06)}`,
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundImage: 'none',
          backgroundColor: alpha('#0A101C', 0.96),
          borderRight: `1px solid ${alpha('#00C8F0', 0.14)}`,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: alpha('#0E1524', 0.92),
          border: `1px solid ${alpha('#00C8F0', 0.14)}`,
          boxShadow: `0 8px 32px ${alpha('#000000', 0.35)}`,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        containedPrimary: {
          background: 'linear-gradient(135deg, #00C8F0 0%, #0090C8 100%)',
          boxShadow: `0 0 16px ${alpha('#00C8F0', 0.35)}`,
          '&:hover': {
            background: 'linear-gradient(135deg, #5CDEFF 0%, #00C8F0 100%)',
            boxShadow: `0 0 22px ${alpha('#00C8F0', 0.5)}`,
          },
        },
        containedSecondary: {
          background: 'linear-gradient(135deg, #FFB020 0%, #E09000 100%)',
        },
      },
    },
    MuiListItem: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          margin: '2px 8px',
          width: 'auto',
          '&.Mui-selected': {
            backgroundColor: alpha('#00C8F0', 0.12),
            borderLeft: `3px solid #00C8F0`,
            '& .MuiListItemIcon-root': {
              color: '#00C8F0',
            },
            '&:hover': {
              backgroundColor: alpha('#00C8F0', 0.18),
            },
          },
        },
      },
    },
    MuiListItemIcon: {
      styleOverrides: {
        root: {
          minWidth: 40,
          color: alpha('#E8F1FF', 0.55),
        },
      },
    },
    MuiTextField: {
      defaultProps: {
        variant: 'outlined',
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-notchedOutline': {
            borderColor: alpha('#00C8F0', 0.22),
          },
          '&:hover .MuiOutlinedInput-notchedOutline': {
            borderColor: alpha('#00C8F0', 0.45),
          },
          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
            borderColor: '#00C8F0',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 6,
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          border: `1px solid ${alpha('#00C8F0', 0.2)}`,
        },
      },
    },
  },
});

export default theme;
