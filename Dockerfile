# 默认 CPU 镜像。如需 GPU，build 时用：
#   docker build --build-arg VARIANT=cuda -t rag-app .
# 注意 GPU 镜像运行时还需 NVIDIA Container Toolkit + `--gpus all`。
ARG VARIANT=cpu

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf_cache \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# 系统依赖：build-essential 给少数从源码编译的包兜底；curl 用于健康检查
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖，利用 Docker layer cache：requirements 不变就不重装
ARG VARIANT
COPY requirements-base.txt requirements-cpu.txt requirements-cuda.txt ./
RUN pip install -r requirements-${VARIANT}.txt

# 再拷代码（变动频繁，放后面）
COPY rag/ ./rag/
COPY app/ ./app/
COPY tests/ ./tests/

# Streamlit 默认 8501
EXPOSE 8501

# 0.0.0.0 让宿主机能访问；headless 关掉首启邮箱提示
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
