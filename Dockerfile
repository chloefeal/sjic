# 前端构建阶段
FROM node:22 as frontend-builder

WORKDIR /app/frontend

# 安装前端依赖
COPY frontend/package*.json ./
RUN npm install

# 构建前端
COPY frontend .
RUN npm run build

# 后端构建阶段
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as backend-builder

WORKDIR /app/backend

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 创建软链接
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# 升级 pip
RUN python -m pip install --no-cache-dir --upgrade pip

# 安装 Python 依赖
COPY backend/requirements.txt .
RUN pip --default-timeout=1000 install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --no-cache-dir -r requirements.txt

# 复制后端代码
COPY backend .

# 编译 Python 文件
RUN python -m compileall -b /app/backend
RUN find /app/backend -name "*.py" -delete

# 最终阶段
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 安装系统依赖和 nginx
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    nginx \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 创建软链接
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# 复制前端构建文件
COPY --from=frontend-builder /app/frontend/build /usr/share/nginx/html

# 复制后端文件和依赖
COPY --from=backend-builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=backend-builder /app/backend /app/backend

# 复制 nginx 配置
COPY nginx.conf /etc/nginx/conf.d/default.conf

WORKDIR /app/backend

# 环境变量
ENV PYTHONPATH=/app/backend
ENV PYTHONDONTWRITEBYTECODE=1
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 启动脚本
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 38880 38881

CMD ["/start.sh"] 
