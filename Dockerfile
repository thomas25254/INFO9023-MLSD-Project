FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY requirements-api.txt .
RUN uv pip install --system -r requirements-api.txt

COPY src/ ./src/

ENV THRESHOLD_PATH="/app/src/threshold_dev-clean.json"
ENV MODEL_PATH="/models/vosk-model-en-us-0.22"
ENV SPK_MODEL_PATH="/artifacts/ecapa_finetuned_speakerid_hidden512.pt"
ENV MODELS_DIR="/models"
ENV ARTIFACTS_DIR="/artifacts"

EXPOSE 8080

WORKDIR /app/src
CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--threads", "8", "--timeout", "600", "app:app"]
