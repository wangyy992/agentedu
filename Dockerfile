# 纯 Python 项目,单阶段即可。刻意不用多阶段 wheel 构建:
# 没有需要编译的自有依赖,多一层只是多一处会坏的地方。
FROM python:3.12-slim

# 不用 root 跑
RUN useradd --create-home --uid 10001 tutor

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[pdf]" \
    && rm -rf /root/.cache

# SQLite 落在 /data。平台挂持久卷到这里就能跨重启保留会话;
# 不挂也能跑,只是重启后进度丢失。
ENV TUTOR_DATA_DIR=/data \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN mkdir -p /data && chown tutor:tutor /data
VOLUME ["/data"]

# 公网部署的默认姿势:关掉按路径读文件(否则是未授权任意文件读取),并给上限。
# 具体数值在平台的环境变量里覆盖。
ENV TUTOR_DEMO_MODE=1 \
    TUTOR_MAX_MATERIAL_CHARS=20000 \
    TUTOR_RATE_LIMIT_PER_MIN=60 \
    TUTOR_INGESTS_PER_HOUR=5 \
    TUTOR_DAILY_LLM_CALL_BUDGET=800

USER tutor
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).status==200 else 1)"

# 单 worker 是有意为之:SQLite 多进程并发写会锁竞争,课程索引也是进程内缓存。
# 要横向扩容得先把存储换成 Postgres(见 README「已知局限」)。
CMD ["uvicorn", "tutor.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
