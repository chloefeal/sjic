import axios from 'axios';

// 导出 getBaseUrl 函数，用于获取基础 URL
export const getBaseUrl = () => {
  // 获取当前访问的域名和协议
  const { protocol, hostname } = window.location;
  if (process.env.NODE_ENV === 'dev') {
    // 开发环境：使用环境变量中的地址
    return process.env.REACT_APP_API_URL;
  } else {
    // 生产环境：使用相同域名，只改变端口
    return `${protocol}//${hostname}:38881`;
  }
};

// 创建 axios 实例
const instance = axios.create({
  baseURL: getBaseUrl(),
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// 请求拦截器
instance.interceptors.request.use(
  config => {
    // 从 localStorage 获取 token
    const token = localStorage.getItem('token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  error => {
    return Promise.reject(error);
  }
);

// 响应拦截器
instance.interceptors.response.use(
  response => {
    return response.data;
  },
  error => {
    if (error.response && error.response.status === 401) {
      // 未授权，清除 token 并重定向到登录页
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// 导出 getWebSocketUrl 函数，用于获取 WebSocket URL
export const getWebSocketUrl = (path) => {
  const baseUrl = getBaseUrl() || window.location.origin;
  const urlObj = new URL(baseUrl);
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${wsProtocol}//${urlObj.host}${path}`;
};

export default instance; 