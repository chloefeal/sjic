# 边缘 Agent 镜像
# Jetson 默认基于 Ultralytics JetPack6 镜像（已含 CUDA / ultralytics / torch）
# 构建示例见同目录 docker-compose.yml

ARG BASE_IMAGE=ultralytics/ultralytics:latest-jetson-jetpack6
FROM ${BASE_IMAGE}

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

# 仅安装 Agent 通用依赖；推理框架由基础镜像提供（Jetson）
# x86 构建时请通过 build-args 换成带 torch 的 requirements
ARG REQUIREMENTS_FILE=requirements.txt
COPY requirements.txt requirements-cpu.txt requirements-jetson.txt ./
RUN pip install --no-cache-dir -r "${REQUIREMENTS_FILE}"

COPY . .

# 运行时目录（也可由 compose volume 覆盖）
RUN mkdir -p /app/models /app/alerts /app/logs

CMD ["python", "main.py"]
