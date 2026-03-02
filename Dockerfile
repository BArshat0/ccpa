FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/opt/venv \
    HF_HOME=/app/.cache/huggingface \
    FORCE_REBUILD_DB=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    build-essential \
    ca-certificates \
    curl \
    git \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:${PATH}"

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

COPY . .

# Optional preloading at build-time so runtime does not need to download models/db.
ARG PRELOAD_ARTIFACTS=0
RUN if [ "${PRELOAD_ARTIFACTS}" = "1" ]; then \
      python download_models.py && \
      python -c "from app import get_vector_db; get_vector_db()"; \
    fi

RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /app/chroma_db /app/.cache/huggingface && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=5 \
  CMD python -c "import http.client,json,sys; c=http.client.HTTPConnection('127.0.0.1',8000,timeout=4); c.request('GET','/health'); r=c.getresponse(); d=json.loads(r.read().decode('utf-8')); sys.exit(0 if ((r.status==200 and d.get('status')=='ok') or (r.status==503 and d.get('status')=='loading')) else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
