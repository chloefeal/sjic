# Build stage
FROM python:3.10 as builder

WORKDIR /app/backend

# 1. 首先安装系统依赖，这部分很少改动
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. 复制并安装 Python 依赖，这部分偶尔改动
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. 最后才复制应用代码，这部分经常改动
COPY backend .

# 编译 Python 文件
RUN python -m compileall -b /app/backend
RUN find /app/backend -name "*.py" -delete

# Final stage
FROM python:3.10-slim

# 1. 首先安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. 复制编译后的文件和依赖
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /app/backend /app/backend

WORKDIR /app/backend

# 3. 设置环境变量
ENV PYTHONPATH=/app/backend
ENV PYTHONDONTWRITEBYTECODE=1

# 运行应用
CMD ["python", "-m", "app"] 