# syntax=docker/dockerfile:1
FROM python:3.11-slim

# git:  needed at runtime -- GitPython shells out to the real `git` binary
# curl: used by the HEALTHCHECK below
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    PORT=8080

WORKDIR /app

# Dependencies first (cached layer). No PyTorch -- uses ONNX for embeddings.
# This keeps the image under 500MB, fitting Render's free tier.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the ONNX embedding model into the image (Chroma's built-in all-MiniLM-L6-v2).
# Downloads ~80 MB once during build, never needed at runtime.
ENV EMBEDDING_THREADS=1 PRELOAD_EMBEDDINGS=true
RUN python -c "from codeanalyzer.embeddings.embedding_manager import get_embedding_model; get_embedding_model()"

COPY . .
RUN pip install -e .

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

# ONE worker process (the analyzed repo lives in that process's memory --
# multiple workers would each have their own copy and "forget" the repo
# between requests), several threads for concurrency, and a generous
# timeout because indexing a large repo can take a few minutes on a small VM.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "600", "app:app"]
