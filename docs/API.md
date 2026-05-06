# HearEdit REST API — Implementation Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [API Endpoints](#api-endpoints)
5. [Implementation Details](#implementation-details)
6. [Docker Packaging](#docker-packaging)
7. [Cloud Deployment (Google Cloud Run)](#cloud-deployment-google-cloud-run)
8. [Environment Variables](#environment-variables)
9. [Key Design Decisions & Challenges](#key-design-decisions--challenges)

---

## Overview

The HearEdit API is a Flask-based REST service that exposes the HearEdit audio transcription and speaker diarization system over HTTP. It accepts audio files (WAV, MP3, etc.), processes them through the full HearEdit pipeline, and returns structured transcription results including per-segment speaker identification and timestamps.

**Base URL (Cloud Run):** `https://hearedit-api-726024632692.europe-west1.run.app`

---

## Architecture

```
Client (browser / curl / any HTTP client)
        │
        │  POST /transcribe (multipart audio file)
        ▼
  Flask API (app.py)
        │
        ├── Saves audio to a temporary file
        ├── Calls transcribe_audio(audio_path)
        │       │
        │       ▼
        │   HearEdit (hearEdit.py)
        │       ├── Transcriber (transcriber.py)
        │       │       ├── FFmpeg  ─── converts audio to 16 kHz mono PCM
        │       │       ├── Vosk    ─── speech-to-text (ASR)
        │       │       └── SpeechBrain ECAPA encoder ─── speaker embeddings
        │       └── Diarizer (diarizer.py)
        │               └── cosine similarity + threshold ─── speaker ID
        │
        └── Returns JSON response with segments + full text
```

The API is stateless per request: each call to `/transcribe` creates a fresh `HearEdit` instance, processes the audio end-to-end, and returns the result. Past transcriptions are stored in memory (`_past_transcriptions` list) and accessible via the `/past_transcriptions` endpoints.

---

## Core Components

### `app.py` — Flask Application

The entry point of the API. Responsibilities:
- Defines all HTTP routes.
- Manages temporary audio file lifecycle (write → process → delete).
- Handles Windows path encoding issues via `_short()`.
- Downloads model files from Google Cloud Storage on startup when `GCS_BUCKET` is set.

**Key helper: `_short(path)`**

On Windows, Vosk's underlying C++ library cannot handle non-ASCII characters in file paths (e.g., accented letters in folder names). This function converts any Windows path to its 8.3 short-path equivalent using the Win32 API:

```python
def _short(path):
    if os.name == "nt":
        import ctypes
        buf = ctypes.create_unicode_buffer(500)
        ctypes.windll.kernel32.GetShortPathNameW(path, buf, 500)
        return buf.value or path
    return path
```

On Linux/Cloud Run this is a no-op.

**Key function: `transcribe_audio(audio_path)`**

Creates a `HearEdit` instance and iterates over all segments until `StopIteration`:

```python
def transcribe_audio(audio_path):
    hear_edit = HearEdit(THRESHOLD_PATH, MODEL_PATH, SPK_MODEL_PATH, audio_path)
    hear_edit.extractor.set_timestamp_format(True)
    try:
        while True:
            hear_edit.play()
    except StopIteration:
        pass
    segments = [
        {
            "start": hear_edit.extracts[eid].start,
            "end":   hear_edit.extracts[eid].end,
            "speaker": hear_edit.extracts[eid].speaker.name
                       if hear_edit.extracts[eid].speaker else "unknown",
            "text": hear_edit.extracts[eid].text(),
        }
        for eid in hear_edit.chronology
    ]
    return segments
```

Segments are collected from `hear_edit.chronology` (the ordered list of extract IDs) after all `play()` calls have completed, ensuring the full audio is processed.

---

### `transcriber.py` — ASR + Speaker Embedding

Wraps Vosk (ASR) and the fine-tuned SpeechBrain ECAPA encoder.

**FFmpeg cross-platform resolution:**

```python
def _get_ffmpeg_path():
    if platform.system() == "Windows":
        import imageio_ffmpeg
        raw = imageio_ffmpeg.get_ffmpeg_exe()
        # Also apply 8.3 short path for Windows
        buf = ctypes.create_unicode_buffer(500)
        ctypes.windll.kernel32.GetShortPathNameW(raw, buf, 500)
        return buf.value if buf.value else raw
    return "ffmpeg"   # Linux: system ffmpeg installed via apt
```

The `Transcriber` class:
- Launches FFmpeg as a subprocess to decode any audio format to 16 kHz, 16-bit, mono PCM.
- Streams the PCM data to Vosk's `KaldiRecognizer` in 4000-byte chunks.
- For each recognized segment, calls `compute_segment_embedding()` which re-runs FFmpeg to extract that time window and passes it to the SpeechBrain ECAPA encoder.
- Returns `Extract` objects containing text, timestamps, and speaker embedding.

**Loading the fine-tuned encoder:**

```python
# In train_speaker_pair_nn.py
def load_finetuned_encoder_only(encoder_path=None):
    sb_encoder = load_sb_encoder(MODEL_DIR)
    path = encoder_path if encoder_path is not None else BEST_ENCODER_PATH
    sb_encoder.mods.embedding_model.load_state_dict(
        torch.load(path, map_location=DEVICE)
    )
    sb_encoder.mods.embedding_model.eval()
    return sb_encoder

def load_sb_encoder(model_dir: str):
    abs_model_dir = os.path.abspath(model_dir)
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",  # valid HF repo ID (bypasses validation)
        savedir=abs_model_dir,                        # local files used if present
    )
    classifier.device = DEVICE
    classifier.mods = classifier.mods.to(DEVICE)
    return classifier
```

> **Note on `source` parameter:** SpeechBrain's `from_hparams()` validates that `source` is a well-formed HuggingFace repository ID. Passing a local directory path directly causes a validation error. The workaround is to pass the real HF repo ID as `source` while pointing `savedir` to the local directory — SpeechBrain checks `savedir` first and skips downloading if the files already exist.

---

### `diarizer.py` — Speaker Identification

Uses cosine similarity between 192-dimensional ECAPA embeddings to assign speakers.

- Each segment's embedding is compared against the mean embedding of all known speakers.
- If the best similarity score is below `threshold` (read from `threshold_dev-clean.json`), a new speaker is created (named `speaker 1`, `speaker 2`, etc.).
- The threshold was calibrated on the LibriSpeech `dev-clean` subset.

---

### `gcs_download.py` — Model Download from GCS

Downloads model files from Google Cloud Storage at container startup. Called once in `app.py` via `_download_models_from_gcs()`.

- **Vosk model** (directory, ~2.7 GB): downloaded using `sync_gcs_prefix_to_dir()` which mirrors the entire GCS prefix to a local directory, preserving sub-folder structure.
- **ECAPA weights** (`.pt` file, ~80 MB): downloaded directly with `blob.download_to_filename()`.

Files already present locally are not re-downloaded (`skip_if_exists=True`).

---

## API Endpoints

### `GET /`

Returns the web UI (HTML page).

**Response:** `200 OK` — HTML page with drag-and-drop upload interface.

---

### `GET /health`

Health check endpoint used by load balancers and monitoring tools.

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /transcribe`

Transcribes an uploaded audio file.

**Request:** `multipart/form-data` with field `audio` containing the audio file.

Supported formats: WAV, MP3, MP4, OGG, FLAC (any format FFmpeg can decode).

**Example (curl):**
```bash
curl -X POST https://hearedit-api-726024632692.europe-west1.run.app/transcribe \
  -F "audio=@debate_extract.wav"
```

**Response `200 OK`:**
```json
{
  "filename": "debate_extract.wav",
  "full_text": "it is only a posteriori through our experience ...",
  "segments": [
    {
      "start": 0.0,
      "end": 20.07,
      "speaker": "speaker 1",
      "text": "it is only a posteriori through our experience of the world ..."
    },
    {
      "start": 20.07,
      "end": 37.5,
      "speaker": "speaker 2",
      "text": "well i think the word necessary ..."
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `filename` | string | Original uploaded filename |
| `full_text` | string | All segments concatenated with spaces |
| `segments` | array | Ordered list of transcription segments |
| `segments[].start` | float | Segment start time in seconds |
| `segments[].end` | float | Segment end time in seconds |
| `segments[].speaker` | string | Identified speaker name (or `"unknown"`) |
| `segments[].text` | string | Transcribed text for this segment |

**Error responses:**

| Status | Body | Cause |
|--------|------|-------|
| `400` | `{"error": "No audio file provided..."}` | Missing `audio` field in form |
| `400` | `{"error": "Empty filename."}` | Empty filename |
| `500` | `{"error": "...", "type": "ExceptionType"}` | Processing error |

---

### `GET /past_transcriptions`

Returns the list of all transcriptions processed since the current instance started.

> **Note:** This is in-memory storage. The list is reset every time the container restarts (scale-to-zero on Cloud Run means this happens frequently).

**Response `200 OK`:**
```json
{
  "count": 2,
  "transcriptions": [
    { "id": 0, "filename": "audio1.wav", "full_text": "..." },
    { "id": 1, "filename": "audio2.wav", "full_text": "..." }
  ]
}
```

---

### `GET /past_transcriptions/<id>`

Returns the full detail of a past transcription by its index.

**Response `200 OK`:** Same structure as `POST /transcribe` response.

**Response `404`:**
```json
{ "error": "Not found" }
```

---

## Implementation Details

### Temporary File Handling

Uploaded audio is saved to a system temporary file, processed, then deleted:

```python
with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
    audio_file.save(tmp.name)
    tmp_path = _short(tmp.name)

segments = transcribe_audio(tmp_path)
# ...
finally:
    if tmp_path and os.path.exists(tmp_path):
        os.unlink(tmp_path)
```

`delete=False` is used because the file must remain accessible after the `with` block closes it (Vosk opens it independently via FFmpeg).

### Web UI (`templates/index.html`)

Single-page interface served at `GET /`:
- Drag-and-drop or click-to-browse file selection.
- Submits the file via `fetch()` as `multipart/form-data`.
- Shows a spinner while waiting.
- Displays full text and a segmented table (segment #, speaker, start, end, text).
- Download button generates a structured `.txt` report client-side using the Blob API — no additional server request needed.

---

## Docker Packaging

The API is containerized using a minimal Python 3.11 slim image.

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies: ffmpeg for audio decoding, gcc/g++ for Python C extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install UV (fast Python package installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python dependencies
COPY requirements-api.txt .
RUN uv pip install --system -r requirements-api.txt

# Copy application source
COPY src/ ./src/

# Model paths inside the container
ENV THRESHOLD_PATH="/app/src/threshold_dev-clean.json"
ENV MODEL_PATH="/models/vosk-model-en-us-0.22"
ENV SPK_MODEL_PATH="/artifacts/ecapa_finetuned_speakerid_hidden512.pt"

EXPOSE 8080

WORKDIR /app/src
CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--threads", "8", "--timeout", "600", "app:app"]
```

**Key choices:**
- `python:3.11-slim` keeps the image size small; system `ffmpeg` is installed via `apt`.
- `uv` is used instead of `pip` for faster dependency installation during builds.
- Models are **not baked into the image** — they are mounted as volumes locally or downloaded from GCS in the cloud. This avoids creating a multi-gigabyte image.
- `--workers 1 --threads 8`: single worker process (PyTorch/Vosk models are not fork-safe) with 8 threads for I/O concurrency.
- `--timeout 600`: allows up to 10 minutes per request (large audio files take time).

### Running Locally with Docker

```bash
# Build
docker build -t hearedit-api .

# Run with local model volumes
docker run -p 5000:8080 \
  -v /path/to/models:/models \
  -v /path/to/artifacts:/artifacts \
  hearedit-api
```

### `requirements-api.txt`

```
flask==3.1.3
gunicorn==23.0.0
vosk==0.3.45
torch==2.11.0
torchaudio==2.11.0
speechbrain==1.0.3
soundfile==0.13.1
huggingface-hub==0.23.4
numpy==2.4.2
requests==2.32.5
scikit-learn
google-cloud-storage
```

> `huggingface-hub` is pinned to `0.23.4` (not the latest). SpeechBrain 1.0.3 uses the deprecated `use_auth_token` parameter of `huggingface_hub`, which was removed in version `1.0+`. Downgrading to `0.23.4` restores compatibility.

---

## Cloud Deployment (Google Cloud Run)

Models are stored in a GCS bucket (`hearedit-models`) and downloaded at container startup.

### GCS Bucket Structure

```
hearedit-models/
├── models/
│   └── vosk-model-en-us-0.22/     (or vosk-model-small-en-us-0.15 for lower memory)
│       ├── am/
│       ├── graph/
│       └── ...
└── artifacts/
    └── ecapa_finetuned_speakerid_hidden512.pt
```

### Deploy Command

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

**Configuration notes:**
- `--min-instances=0`: scale-to-zero when idle to minimize cost.
- `--max-instances=1`: prevents parallel instances (model loading is expensive; only one instance at a time).
- `--memory=16Gi`: required because the full Vosk model (~2.7 GB in RAM) + PyTorch/SpeechBrain (~4–5 GB) exceeds 8 GB. An alternative is to use `vosk-model-small-en-us-0.15` (~40 MB) which brings total RAM under 8 GB.
- `--timeout=600`: matches the gunicorn timeout.
- `GCS_BUCKET`: triggers automatic model download at startup.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Port for the Flask dev server (not used with gunicorn) |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |
| `THRESHOLD_PATH` | `src/threshold_dev-clean.json` | Path to the diarization threshold JSON |
| `MODEL_PATH` | `../models/vosk-model-en-us-0.22` | Path to the Vosk ASR model directory |
| `SPK_MODEL_PATH` | `../artifacts/ecapa_finetuned_speakerid_hidden512.pt` | Path to the fine-tuned ECAPA weights |
| `MODELS_DIR` | `../models` | Root directory for model downloads |
| `ARTIFACTS_DIR` | `../artifacts` | Root directory for artifact downloads |
| `GCS_BUCKET` | _(empty)_ | GCS bucket name; if set, models are downloaded at startup |
| `GCS_MODELS_PREFIX` | `models/vosk-model-en-us-0.22` | GCS prefix for the Vosk model |
| `GCS_ARTIFACT_PREFIX` | `artifacts/ecapa_finetuned_speakerid_hidden512.pt` | GCS path for the ECAPA weights |
