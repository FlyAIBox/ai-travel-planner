FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt /app/
COPY pyproject.toml /app/

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 下载必要的模型和数据
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
RUN python -c "import spacy; spacy.cli.download('en_core_web_sm')"

# 创建数据目录
RUN mkdir -p /app/data/logs /app/data/backups /app/data/knowledge_versions

# 复制应用代码
COPY services/chat-service/ /app/
COPY shared/ /app/shared/

# 设置环境变量
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production
ENV LOG_LEVEL=info

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"] 