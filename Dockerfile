FROM python:3.11-slim

WORKDIR /app

# System dependencies: ffmpeg for audio, gcc/g++ for Python packages that compile C extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python dependencies
COPY requirements-api.txt .
RUN uv pip install --system -r requirements-api.txt

# Copy source code (contains app.py, all modules, templates/, and Ntrain/models/)
COPY src/ ./src/

# Vosk model and finetuned .pt weights are large — mount them at runtime via volumes.
# Set default paths inside the container (matching the volume mount points below).
ENV THRESHOLD_PATH="/app/src/threshold_dev-clean.json"
ENV MODEL_PATH="/models/vosk-model-en-us-0.22"
ENV SPK_MODEL_PATH="/artifacts/ecapa_finetuned_speakerid_hidden512.pt"

EXPOSE 8080

# Run from src/ so relative imports and template discovery work correctly
WORKDIR /app/src
CMD ["python", "app.py"]
