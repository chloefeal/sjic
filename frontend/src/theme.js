import { createTheme, alpha } from '@mui/material/styles';

export const DEFAULT_UI_THEME = 'night-tech';

export const UI_THEME_OPTIONS = [
  {
    value: 'night-tech',
    label: '科技夜色',
    description: '青蓝网格，适合夜间监考大屏',
    preview: 'linear-gradient(135deg, #050912 0%, #083044 55%, #2A1C08 100%)',
  },
  {
    value: 'exam-dawn',
    label: '晨光考场',
    description: '暖金日出，贴近科目二场地氛围',
    preview: 'linear-gradient(135deg, #1C1208 0%, #C47A22 50%, #E07A3A 100%)',
  },
  {
    value: 'highway-navy',
    label: '深蓝车道',
    description: '夜航公路灯光，沉稳专业',
    preview: 'linear-gradient(135deg, #041018 0%, #0B3D5C 50%, #146B4A 100%)',
  },
  {
    value: 'proctor-light',
    label: '明亮监考',
    description: '浅色考场风格，日间办公更清晰',
    preview: 'linear-gradient(135deg, #D9E6F5 0%, #F7FBFF 55%, #F3E4C4 100%)',
  },
];

const TOKENS = {
  'night-tech': {
    mode: 'dark',
    primary: '#00C8F0',
    secondary: '#FFB020',
    background: '#070B14',
    paper: '#0E1524',
    text: '#E8F1FF',
    textSecondary: '#8BA0BC',
    appBar: '#0A101C',
    titleGradient: 'linear-gradient(90deg, #E8F1FF 0%, #00C8F0 55%, #FFB020 100%)',
  },
  'exam-dawn': {
    mode: 'dark',
    primary: '#E8A03A',
    secondary: '#FF6B3D',
    background: '#140E08',
    paper: '#1E140C',
    text: '#FFF4E5',
    textSecondary: '#C4A882',
    appBar: '#1A120A',
    titleGradient: 'linear-gradient(90deg, #FFF4E5 0%, #E8A03A 55%, #FF6B3D 100%)',
  },
  'highway-navy': {
    mode: 'dark',
    primary: '#3DDC97',
    secondary: '#5B9CFF',
    background: '#051018',
    paper: '#0A1A24',
    text: '#E6F4FF',
    textSecondary: '#7FA3B8',
    appBar: '#07141C',
    titleGradient: 'linear-gradient(90deg, #E6F4FF 0%, #5B9CFF 50%, #3DDC97 100%)',
  },
  'proctor-light': {
    mode: 'light',
    primary: '#1565C0',
    secondary: '#C9A227',
    background: '#F3F6FA',
    paper: '#FFFFFF',
    text: '#1A2332',
    textSecondary: '#5A6A80',
    appBar: '#FFFFFF',
    titleGradient: 'linear-gradient(90deg, #0D47A1 0%, #1565C0 55%, #C9A227 100%)',
  },
};

export const LOGIN_VISUALS = {
  'night-tech': {
    pageBg: '#050912',
    backgroundImage: (a) => `
      radial-gradient(ellipse 70% 55% at 15% 20%, ${a('#00C8F0', 0.22)}, transparent 55%),
      radial-gradient(ellipse 50% 45% at 85% 75%, ${a('#FFB020', 0.14)}, transparent 50%),
      radial-gradient(ellipse 40% 35% at 70% 15%, ${a('#00C8F0', 0.1)}, transparent 45%),
      linear-gradient(${a('#00C8F0', 0.05)} 1px, transparent 1px),
      linear-gradient(90deg, ${a('#00C8F0', 0.05)} 1px, transparent 1px)
    `,
    backgroundSize: 'auto, auto, auto, 56px 56px, 56px 56px',
    orb1: '#00C8F0',
    orb2: '#FFB020',
    scan: '#00C8F0',
    cardBg: a => a('#0A1220', 0.72),
    cardBorder: a => a('#00C8F0', 0.28),
    cardGlow: a => a('#00C8F0', 0.12),
    accent: a => a('#FFB020', 0.08),
    footer: a => a('#E8F1FF', 0.45),
    showScan: true,
  },
  'exam-dawn': {
    pageBg: '#120C08',
    backgroundImage: (a) => `
      radial-gradient(ellipse 90% 55% at 50% 115%, ${a('#FF8A3D', 0.38)}, transparent 55%),
      radial-gradient(ellipse 50% 40% at 12% 18%, ${a('#E8A03A', 0.22)}, transparent 50%),
      linear-gradient(180deg, ${a('#1C1208', 0.2)} 0%, transparent 40%),
      linear-gradient(${a('#E8A03A', 0.06)} 1px, transparent 1px),
      linear-gradient(90deg, ${a('#E8A03A', 0.06)} 1px, transparent 1px)
    `,
    backgroundSize: 'auto, auto, auto, 64px 64px, 64px 64px',
    orb1: '#E8A03A',
    orb2: '#FF6B3D',
    scan: '#E8A03A',
    cardBg: a => a('#1A120C', 0.78),
    cardBorder: a => a('#E8A03A', 0.32),
    cardGlow: a => a('#FF6B3D', 0.14),
    accent: a => a('#FF6B3D', 0.1),
    footer: a => a('#FFF4E5', 0.5),
    showScan: false,
  },
  'highway-navy': {
    pageBg: '#030B12',
    backgroundImage: (a) => `
      radial-gradient(ellipse 40% 80% at 50% 100%, ${a('#5B9CFF', 0.18)}, transparent 60%),
      repeating-linear-gradient(90deg, transparent 0 46px, ${a('#E8F4FF', 0.06)} 46px 48px, transparent 48px 96px),
      linear-gradient(180deg, transparent 58%, ${a('#041018', 0.9) } 100%),
      radial-gradient(ellipse 60% 40% at 80% 20%, ${a('#3DDC97', 0.12)}, transparent 50%)
    `,
    backgroundSize: 'auto, 96px 100%, auto, auto',
    orb1: '#5B9CFF',
    orb2: '#3DDC97',
    scan: '#5B9CFF',
    cardBg: a => a('#07141C', 0.82),
    cardBorder: a => a('#3DDC97', 0.28),
    cardGlow: a => a('#5B9CFF', 0.14),
    accent: a => a('#3DDC97', 0.1),
    footer: a => a('#E6F4FF', 0.45),
    showScan: true,
  },
  'proctor-light': {
    pageBg: '#EAF0F7',
    backgroundImage: (a) => `
      radial-gradient(ellipse 80% 50% at 50% -10%, ${a('#90CAF9', 0.55)}, transparent 55%),
      radial-gradient(ellipse 50% 40% at 90% 90%, ${a('#C9A227', 0.16)}, transparent 50%),
      linear-gradient(${a('#1565C0', 0.05)} 1px, transparent 1px),
      linear-gradient(90deg, ${a('#1565C0', 0.05)} 1px, transparent 1px)
    `,
    backgroundSize: 'auto, auto, 48px 48px, 48px 48px',
    orb1: '#90CAF9',
    orb2: '#C9A227',
    scan: '#1565C0',
    cardBg: a => a('#FFFFFF', 0.88),
    cardBorder: a => a('#1565C0', 0.18),
    cardGlow: a => a('#1565C0', 0.08),
    accent: a => a('#C9A227', 0.12),
    footer: a => a('#1A2332', 0.45),
    showScan: false,
  },
};

export function createAppTheme(themeId = DEFAULT_UI_THEME) {
  const t = TOKENS[themeId] || TOKENS[DEFAULT_UI_THEME];
  const isLight = t.mode === 'light';

  return createTheme({
    palette: {
      mode: t.mode,
      primary: {
        main: t.primary,
        contrastText: isLight ? '#FFFFFF' : '#031018',
      },
      secondary: {
        main: t.secondary,
        contrastText: isLight ? '#1A1200' : '#1A1200',
      },
      background: {
        default: t.background,
        paper: t.paper,
      },
      text: {
        primary: t.text,
        secondary: t.textSecondary,
      },
      divider: alpha(t.primary, isLight ? 0.16 : 0.12),
      success: { main: '#22C55E' },
      warning: { main: t.secondary },
      error: { main: isLight ? '#D32F2F' : '#FF5C7A' },
      info: { main: t.primary },
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
              radial-gradient(ellipse 80% 50% at 10% -10%, ${alpha(t.primary, isLight ? 0.14 : 0.12)}, transparent 55%),
              radial-gradient(ellipse 60% 40% at 100% 0%, ${alpha(t.secondary, isLight ? 0.1 : 0.06)}, transparent 50%),
              linear-gradient(${alpha(t.primary, 0.03)} 1px, transparent 1px),
              linear-gradient(90deg, ${alpha(t.primary, 0.03)} 1px, transparent 1px)
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
            backgroundColor: alpha(t.appBar, isLight ? 0.92 : 0.85),
            backdropFilter: 'blur(12px)',
            borderBottom: `1px solid ${alpha(t.primary, 0.18)}`,
            boxShadow: `0 0 24px ${alpha(t.primary, 0.06)}`,
            color: t.text,
          },
        },
      },
      MuiDrawer: {
        styleOverrides: {
          paper: {
            backgroundImage: 'none',
            backgroundColor: alpha(t.appBar, isLight ? 0.98 : 0.96),
            borderRight: `1px solid ${alpha(t.primary, 0.14)}`,
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            backgroundImage: 'none',
            backgroundColor: alpha(t.paper, isLight ? 1 : 0.92),
            border: `1px solid ${alpha(t.primary, isLight ? 0.1 : 0.14)}`,
            boxShadow: `0 8px 32px ${alpha('#000000', isLight ? 0.08 : 0.35)}`,
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          containedPrimary: {
            background: `linear-gradient(135deg, ${t.primary} 0%, ${t.primary} 100%)`,
            boxShadow: `0 0 16px ${alpha(t.primary, 0.35)}`,
            '&:hover': {
              boxShadow: `0 0 22px ${alpha(t.primary, 0.5)}`,
            },
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
              backgroundColor: alpha(t.primary, 0.12),
              borderLeft: `3px solid ${t.primary}`,
              '& .MuiListItemIcon-root': {
                color: t.primary,
              },
              '&:hover': {
                backgroundColor: alpha(t.primary, 0.18),
              },
            },
          },
        },
      },
      MuiListItemIcon: {
        styleOverrides: {
          root: {
            minWidth: 40,
            color: alpha(t.text, 0.55),
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
              borderColor: alpha(t.primary, 0.22),
            },
            '&:hover .MuiOutlinedInput-notchedOutline': {
              borderColor: alpha(t.primary, 0.45),
            },
            '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
              borderColor: t.primary,
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
            border: `1px solid ${alpha(t.primary, 0.2)}`,
          },
        },
      },
    },
  });
}

export function getLoginVisual(themeId = DEFAULT_UI_THEME) {
  return LOGIN_VISUALS[themeId] || LOGIN_VISUALS[DEFAULT_UI_THEME];
}

export function getTitleGradient(themeId = DEFAULT_UI_THEME) {
  const t = TOKENS[themeId] || TOKENS[DEFAULT_UI_THEME];
  return t.titleGradient;
}

const theme = createAppTheme(DEFAULT_UI_THEME);
export default theme;
