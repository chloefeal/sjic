import axios from 'axios';

const getBaseUrl = () => {
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

const instance = axios.create({
  baseURL: getBaseUrl()
});

// 请求拦截器
instance.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器
instance.interceptors.response.use(
  (response) => response.data,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default instance; 