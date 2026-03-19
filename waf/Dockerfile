FROM python:3.10-slim

RUN groupadd -g 1000 wafservice && \
    useradd -m -u 1000 -g wafservice -s /usr/sbin/nologin wafservice && \
    mkdir -p /home/wafservice /tmp/torch_cache /tmp/torch_home && \
    chown -R 1000:1000 /home/wafservice /tmp/torch_cache /tmp/torch_home

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY . .
RUN chown -R 1000:1000 /app

ENV PYTHONUNBUFFERED=1
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache
ENV TORCH_HOME=/tmp/torch_home
ENV TORCH_COMPILE_DISABLE=1
ENV HOME=/home/wafservice

USER 1000:1000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "65", "--log-level", "info"]
