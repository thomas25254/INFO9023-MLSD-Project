# Deployment — Docker & Google Cloud Run

## Overview

The HearEdit API is packaged as a Docker container and deployed on **Google Cloud Run** — a fully managed serverless platform. The container starts on demand and scales to zero when idle, minimizing cost.

**Live URL:** `https://hearedit-api-726024632692.europe-west1.run.app`

---

## Architecture

```
Developer machine
      │
      │  gcloud run deploy --source=$(pwd)
      ▼
Cloud Build  ──►  Artifact Registry  ──►  Cloud Run (europe-west1)
                                               │
                                        on first request:
                                        download models from GCS
                                               │
                                        gs://hearedit-models/
                                        ├── models/vosk-model-en-us-0.22/
                                        └── artifacts/ecapa_finetuned_speakerid_hidden512.pt
```

Models are **not baked into the image**. They are stored in GCS and downloaded to the container's filesystem on the first request after startup. This keeps the Docker image lightweight (~2 GB instead of ~5 GB) and allows model updates without rebuilding the image.

---

## Docker Image

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System packages: ffmpeg for audio decoding, gcc/g++ for Python C extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# UV: fast Python package installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY requirements-api.txt .
RUN uv pip install --system -r requirements-api.txt

COPY src/ ./src/

ENV THRESHOLD_PATH="/app/src/threshold_dev-clean.json"
ENV MODEL_PATH="/models/vosk-model-en-us-0.22"
ENV SPK_MODEL_PATH="/artifacts/ecapa_finetuned_speakerid_hidden512.pt"

EXPOSE 8080

WORKDIR /app/src
CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--threads", "8", "--timeout", "600", "app:app"]
```

### Key design choices

| Choice | Reason |
|--------|--------|
| `python:3.11-slim` | Minimal base image; system `ffmpeg` installed via `apt` instead of `imageio-ffmpeg` |
| `uv` instead of `pip` | Significantly faster dependency installation during Cloud Build |
| `--workers 1` | PyTorch and Vosk models are not fork-safe; a single worker avoids duplicating ~8 GB of model state across processes |
| `--threads 8` | Allows I/O concurrency within the single worker process |
| `--timeout 600` | Large audio files can take several minutes to transcribe |
| Models via GCS, not COPY | Keeps image small; avoids rebuilding the image when model weights change |

### `requirements-api.txt`

```
flask==3.1.3
gunicorn==23.0.0
vosk==0.3.45
torch==2.11.0
torchaudio==2.11.0
speechbrain==1.0.3
soundfile==0.13.1
huggingface-hub==0.23.4      # pinned: SpeechBrain 1.0.3 uses deprecated use_auth_token
numpy==2.4.2
requests==2.32.5
scikit-learn
google-cloud-storage
```

> `huggingface-hub` is pinned to `0.23.4`. SpeechBrain 1.0.3 calls `use_auth_token` internally, which was removed in `huggingface-hub >= 1.0`. Upgrading huggingface-hub breaks the SpeechBrain model loading.

---

## Running Locally with Docker

### Build

```bash
docker build -t hearedit-api .
```

### Run (with local model volumes)

```bash
docker run -p 5000:8080 \
  -v $(pwd)/models:/models \
  -v $(pwd)/artifacts:/artifacts \
  hearedit-api
```

- The container listens on port **8080**; it is mapped to **5000** on the host.
- Models must be available in `./models/` and `./artifacts/` on the host, or the `GCS_BUCKET` env var must be set.

### Test locally

```bash
# Health check
curl http://localhost:5000/health

# Transcribe
curl -X POST http://localhost:5000/transcribe -F "audio=@debate_extract.wav"
```

---

## GCS Model Storage

Models are stored in a dedicated bucket separate from the training data bucket:

```
gs://hearedit-models/
├── models/
│   └── vosk-model-en-us-0.22/     # Vosk ASR model (~2.7 GB directory)
│       ├── am/
│       ├── graph/
│       └── ...
└── artifacts/
    └── ecapa_finetuned_speakerid_hidden512.pt   # Fine-tuned ECAPA weights (~80 MB)
```

### Upload commands

```bash
# Upload the Vosk model directory
gsutil -m cp -r models/vosk-model-en-us-0.22 gs://hearedit-models/models/

# Upload the fine-tuned ECAPA weights
gsutil cp artifacts/ecapa_finetuned_speakerid_hidden512.pt gs://hearedit-models/artifacts/
```

### Download logic in the container

At startup, `app.py` calls `_download_models_from_gcs()` which:

1. **Vosk model**: uses `sync_gcs_prefix_to_dir()` to mirror the GCS prefix into `MODEL_PATH`. Already-present files are skipped (`skip_if_exists=True`).
2. **ECAPA weights**: uses `blob.download_to_filename(SPK_MODEL_PATH)` directly — a single file blob.

If `GCS_BUCKET` is not set, this function is a no-op and models must be present at the configured paths.

---

## Deploying to Cloud Run

### Prerequisites

```bash
# Authenticate
gcloud auth login
gcloud config set project info9023-project-hearedit

# Enable required APIs (one-time)
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

### Deploy command

```bash
gcloud run deploy hearedit-api \
  --project=info9023-project-hearedit \
  --region=europe-west1 \
  --source=$(pwd) \
  --memory=16Gi \
  --cpu=4 \
  --timeout=600 \
  --min-instances=0 \
  --max-instances=1 \
  --set-env-vars="GCS_BUCKET=hearedit-models" \
  --allow-unauthenticated
```

### Configuration parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `--region=europe-west1` | Belgium | Close to Liège; GCS bucket is in the same region |
| `--memory=16Gi` | 16 GB | Vosk (~2.7 GB) + PyTorch/SpeechBrain (~4–5 GB) + overhead exceeds 8 GB |
| `--cpu=4` | 4 vCPU | Required by Cloud Run when `--memory > 8Gi` |
| `--timeout=600` | 10 min | Large audio files take several minutes |
| `--min-instances=0` | 0 | Scale to zero when idle — no cost when not in use |
| `--max-instances=1` | 1 | Prevents concurrent model loading (each instance loads ~8 GB) |
| `--allow-unauthenticated` | — | Public API, no auth token required |

### Environment variables set at deploy time

| Variable | Value |
|----------|-------|
| `GCS_BUCKET` | `hearedit-models` |
| `THRESHOLD_PATH` | `/app/src/threshold_dev-clean.json` (built into image) |
| `MODEL_PATH` | `/models/vosk-model-en-us-0.22` (downloaded from GCS) |
| `SPK_MODEL_PATH` | `/artifacts/ecapa_finetuned_speakerid_hidden512.pt` (downloaded from GCS) |

---

## Memory Considerations

The total RAM usage at inference time is approximately:

| Component | RAM usage |
|-----------|-----------|
| Vosk `vosk-model-en-us-0.22` | ~2.7 GB |
| PyTorch + SpeechBrain ECAPA | ~4–5 GB |
| Python / gunicorn overhead | ~0.5 GB |
| **Total** | **~7.5–8.5 GB** |

This exceeds the 8 Gi Cloud Run limit, hence the `--memory=16Gi` requirement.

### Alternative: smaller Vosk model

Using `vosk-model-small-en-us-0.15` (~40 MB) reduces total RAM to ~4–5 GB, fitting within an 8 Gi instance:

```bash
# Upload small model
gsutil -m cp -r vosk-model-small-en-us-0.15 gs://hearedit-models/models/

# Deploy with small model and reduced memory
gcloud run deploy hearedit-api \
  --memory=8Gi \
  --cpu=2 \
  --set-env-vars="GCS_BUCKET=hearedit-models,\
GCS_MODELS_PREFIX=models/vosk-model-small-en-us-0.15,\
MODEL_PATH=/models/vosk-model-small-en-us-0.15"
```

Transcription quality is lower with the small model, but it is adequate for demonstration purposes.

---

## Cost Management

Cloud Run charges only for actual request processing time (not idle time, since `--min-instances=0`).

Key settings to minimize cost:
- `--min-instances=0`: container shuts down when no traffic → $0 when not in use.
- `--max-instances=1`: prevents over-scaling; model inference is not horizontally parallelizable at this stage.
- Keep `--timeout` matched to gunicorn's `--timeout` to avoid orphaned requests being charged.

> Monitor usage at: [console.cloud.google.com/run](https://console.cloud.google.com/run)

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Memory limit of 8192 MiB exceeded` | Model + PyTorch > 8 GB | Use `--memory=16Gi` or switch to small Vosk model |
| `FileNotFoundError: /artifacts/...pt` | ECAPA weights not in GCS or wrong path | Check `gs://hearedit-models/artifacts/` and `GCS_ARTIFACT_PREFIX` env var |
| `Failed to create model [Vosk]` | Model dir missing or wrong path | Check `gs://hearedit-models/models/` and `GCS_MODELS_PREFIX` env var |
| `Repo id must be in form 'repo_name'` | SpeechBrain `source` is a local path | Ensure `load_sb_encoder` uses `source="speechbrain/spkrec-ecapa-voxceleb"` with `savedir=local_path` |
| `use_auth_token` error | `huggingface-hub >= 1.0` | Pin `huggingface-hub==0.23.4` in `requirements-api.txt` |
| Container starts but `/transcribe` returns 503 | OOM during model loading | Increase `--memory` |
