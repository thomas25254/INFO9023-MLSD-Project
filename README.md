# HearEdit

> **INFO9023 — Machine Learning Systems Design**

HearEdit is an automatic **speaker diarization and transcription** pipeline. Given an audio recording of a conversation, it identifies *who spoke when*, transcribes the speech, and assigns each segment to a speaker — including speakers never seen during training.

A **REST API** wraps the pipeline and is deployed on **Google Cloud Run**, making it callable from any machine over HTTP : https://hearedit-api-726024632692.europe-west1.run.app/

---

## Table of Contents

- [Use Case](#use-case)
- [Project Structure](#project-structure)
- [System Overview](#system-overview)
- [Getting Started](#getting-started)
- [Data](#data)
- [ML Models](#ml-models)
- [Pipeline](#pipeline)
- [REST API](#rest-api)
- [Monitoring & Dashboard](#monitoring--dashboard)
- [CI/CD](#cicd)
- [Documentation](#documentation)

---

## Use Case

Many real-world recordings — meetings, interviews, podcasts, lectures — involve multiple speakers. HearEdit automatically segments and labels those recordings by speaker identity, producing a structured transcript with:

- Per-segment **speaker labels** (e.g. `speaker 1`, `speaker 2`, or user-assigned names)
- **Timestamps** (start and end time in seconds for each segment)
- **Transcribed text** per segment

The system is **zero-shot**: no speaker-specific training is needed at inference time. Speakers are identified by comparing voice embeddings against known prototypes using cosine similarity and a calibrated threshold.

---

## Project Structure

```
INFO9023-MLSD-Project/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                   # CI pipeline: pre-commit + pytest
│       └── deploy.yml               # CD pipeline: Docker build & Cloud Run deploy
│
├── src/
│   ├── app.py                       # Flask REST API entry point
│   ├── hearEdit.py                  # HearEdit orchestrator (main pipeline class)
│   ├── transcriber.py               # ASR (Vosk) + speaker embedding (ECAPA)
│   ├── diarizer.py                  # Speaker identification via cosine similarity
│   ├── speaker.py                   # Speaker prototype management
│   ├── extractor.py                 # Extract (segment) data model
│   ├── speaker_embedding_threshold.py  # Threshold calibration script (Milestone 1)
│   ├── train_speaker_pair_nn.py     # ECAPA fine-tuning script (Milestone 2)
│   ├── gcs_download.py              # GCS → local file sync utility
│   ├── utils.py                     # Pure utility functions (normalize, IQR, EER…)
│   ├── threshold_dev-clean.json     # Calibrated cosine similarity threshold
│   ├── speakers_embeddings_dev-clean.npz  # Cached speaker embeddings
│   ├── gcs_upload.ipynb             # One-time GCS upload notebook
│   ├── vosk_speaker_recognition_bs.ipynb  # EDA & prototyping notebook
│   ├── templates/
│   │   └── index.html               # Web UI served at GET /
│   ├── pipeline/                    # Vertex AI KFP training pipeline
│   │   ├── run_pipeline.py          # Compile & submit the pipeline to Vertex AI
│   │   ├── config.py                # Pipeline configuration (URIs, hyperparameters)
│   │   ├── data_preparation.py      # KFP component: download & split LibriSpeech
│   │   ├── training.py              # KFP component: ECAPA fine-tuning (GPU)
│   │   ├── evaluation.py            # KFP component: EER / AUC on test set
│   │   ├── threshold.py             # KFP component: compute production threshold
│   │   ├── hearedit_pipeline.json   # Compiled KFP pipeline definition
│   │   ├── requirements-pipeline.txt
│   │   ├── Dockerfile               # Base image for all KFP components
│   │   └── test_local.py            # Local smoke-test (no Vertex AI)
│   ├── streamlit/                   # Monitoring dashboard
│   │   ├── dashboard.py             # Streamlit app (transcribe + history pages)
│   │   ├── main.py                  # Entry point
│   │   ├── Dockerfile               # Container for Cloud Run deployment
│   │   └── pyproject.toml           # Dashboard dependencies (uv)
│   └── Ntrain/
│       ├── models/                  # SpeechBrain model files (used during fine-tuning)
│       └── train_speaker_pair_nn.py # Legacy training entry point
│
├── notebooks/
│   └── model_training/
│       └── vosk_speaker_recognition/  # Exploration notebooks
│
├── tests/
│   └── test_utils.py                # Unit tests for pure ML logic
│
├── docs/
│   ├── API.md                       # REST API full documentation
│   ├── DATA.md                      # Cloud storage design decisions
│   ├── DEPLOYMENT.md                # Docker & Cloud Run deployment guide
│   ├── EXPERIMENTATION.md           # Threshold & fine-tuning results
│   ├── PIPELINE.md                  # Vertex AI KFP pipeline documentation
│   ├── MONITORING.md                # Streamlit dashboard & BigQuery monitoring
│   ├── SPEAKER_ENCODER.md           # Explain the train on speaker encoder
│   └── pictures/                    # Plots referenced by docs
│
├── slides/
│   ├── milestone_1.pdf
│   └── milestone_2.pdf
│
├── Dockerfile                       # Container definition for the API
├── requirements.txt                 # Dev/training dependencies
├── requirements-api.txt             # API/Docker runtime dependencies
├── ruff.toml                        # Linter configuration
├── .pre-commit-config.yaml          # Pre-commit hooks
├── .gitignore
```

---

## System Overview

```
Audio file (WAV / MP3 / FLAC / …)
        │
        ▼
   Transcriber
   ├── FFmpeg      → decode to 16 kHz mono PCM
   ├── Vosk        → speech-to-text (ASR), produces word-level timestamps
   └── ECAPA       → 192-dim voice embedding per segment
        │
        ▼
    Diarizer
   └── cosine similarity vs. known speaker prototypes
       ├── similarity > threshold  →  assign to existing speaker
       └── similarity ≤ threshold  →  create new speaker
        │
        ▼
   HearEdit (orchestrator)
   └── chronology of Extract objects
       {start, end, speaker, text, embedding}
        │
        ▼
   Flask API  →  JSON response  →  Client
```

**Two ML models are involved:**

| Model | Role | Training |
|-------|------|----------|
| Vosk `vosk-model-en-us-0.22` | ASR — speech-to-text | Pre-trained, not fine-tuned |
| SpeechBrain ECAPA-TDNN | Speaker embeddings | Pre-trained on VoxCeleb, fine-tuned on LibriSpeech `train-clean-100` |

The diarization threshold (`0.437`) is calibrated on LibriSpeech `dev-clean` using the **Equal Error Rate (EER)** criterion — see [`docs/EXPERIMENTATION.md`](docs/EXPERIMENTATION.md).

---

## Getting Started

### Prerequisites

- Python 3.11
- FFmpeg (Linux: `apt install ffmpeg` / Windows: `imageio-ffmpeg` via pip)
- The Vosk model and fine-tuned ECAPA weights (see below)

### Install dependencies

```bash
pip install -r requirements.txt        # development / training
pip install -r requirements-api.txt    # API runtime (also used in Docker)
```

### Download models

| Model | Source | Local path |
|-------|--------|------------|
| `vosk-model-en-us-0.22` | [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) | `models/vosk-model-en-us-0.22/` |
| ECAPA fine-tuned weights | GCS: `gs://hearedit-models/artifacts/` | `artifacts/ecapa_finetuned_speakerid_hidden512.pt` |

### Run the API locally

```bash
cd src
python app.py
# → http://localhost:5000
```

Or with Docker (models mounted as volumes):

```bash
docker build -t hearedit-api .
docker run -p 5000:8080 \
  -v $(pwd)/models:/models \
  -v $(pwd)/artifacts:/artifacts \
  hearedit-api
```

### Run the tests

```bash
PYTHONPATH=src pytest tests/ -v
```

---

## Data

The primary dataset is **LibriSpeech** — read English audiobooks:

| Split | Purpose | Speakers | Hours |
|-------|---------|----------|-------|
| `dev-clean` | Threshold calibration | 40 | ~5 h |
| `train-clean-100` | ECAPA fine-tuning | 251 | ~100 h |

Data is stored on **Google Cloud Storage**:

```
gs://mlops-2026-dataset-bucket/dev-clean/LibriSpeech/      # threshold calibration
```

Models and fine-tuned weights are stored separately:

```
gs://hearedit-models/
├── models/vosk-model-en-us-0.22/
└── artifacts/ecapa_finetuned_speakerid_hidden512.pt
```

Full storage design rationale → [`docs/DATA.md`](docs/DATA.md)

> `data/`, `models/`, and `artifacts/` are in `.gitignore` and never committed.

---

## ML Models

### Milestone 1 — Threshold Calibration

The original speaker model used Vosk's built-in speaker embeddings (`vosk-model-spk-0.4`). A cosine similarity threshold was calibrated on LibriSpeech `dev-clean` using the **EER criterion**:

| Metric | Value |
|--------|-------|
| AUC | 0.9683 |
| EER | 9.50% |
| Threshold | 0.437 |

### Milestone 2 — ECAPA Fine-tuning

The speaker embedding model was upgraded to **SpeechBrain's ECAPA-TDNN**, fine-tuned on LibriSpeech `train-clean-100` as a speaker classification task:

- **Architecture**: ECAPA-TDNN encoder + linear classifier head (hidden dim 512)
- **Training**: 10 epochs, encoder LR 3×10⁻⁶, classifier LR 5×10⁻⁴, batch size 16
- **Output**: 192-dim speaker embeddings replacing the original Vosk embeddings
- **Threshold**: recalibrated on `dev-clean` using the same EER methodology

Full methodology, training curves, and results → [`docs/EXPERIMENTATION.md`](docs/EXPERIMENTATION.md)

---

## Pipeline

The retraining lifecycle is automated with a **Vertex AI Kubeflow Pipelines (KFP)** pipeline defined in [`src/pipeline/`](src/pipeline/). It orchestrates four components that run entirely in the cloud without manual intervention:

```
data_preparation (CPU)
        │
        ├── train_split ──┐
        ├── val_split   ──┤──► training (GPU T4)
        └── test_split  ──┤         │
                          │         ├── model ──► evaluation (CPU)
                          │         └── model ──► compute_threshold (CPU)
                          └─────────┘
```

| Component | Machine | What it does |
|-----------|---------|--------------|
| `data_preparation` | CPU | Downloads LibriSpeech from GCS, splits by speaker (80/10/10) |
| `training` | n1-standard-4 + T4 GPU | Fine-tunes ECAPA-TDNN with ArcFace loss, saves best checkpoint |
| `evaluation` | CPU | Computes EER & AUC on the held-out test set |
| `compute_threshold` | CPU | Derives the production cosine-similarity threshold (EER criterion) |

The output `threshold_pipeline.json` is written directly to GCS and loaded by the Flask API at startup — no redeployment needed.

### Run the pipeline

```bash
cd src/pipeline
python run_pipeline.py   # compiles + submits to Vertex AI
```

Monitor execution in **GCP Console → Vertex AI → Pipelines → Runs** (region: `europe-west1`).

Full pipeline documentation → [`docs/PIPELINE.md`](docs/PIPELINE.md)

---

## REST API

The pipeline is served as a REST API on **Google Cloud Run**:

```
https://hearedit-api-726024632692.europe-west1.run.app
```

### Quick example

```bash
curl -X POST https://hearedit-api-726024632692.europe-west1.run.app/transcribe \
  -F "audio=@my_recording.wav"
```

**Response:**
```json
{
  "filename": "my_recording.wav",
  "full_text": "it is only a posteriori through our experience ...",
  "segments": [
    { "start": 0.0, "end": 20.07, "speaker": "speaker 1", "text": "it is only..." },
    { "start": 20.07, "end": 37.5, "speaker": "speaker 2", "text": "well I think..." }
  ]
}
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI (drag-and-drop upload) |
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/transcribe` | Transcribe an audio file |
| `GET` | `/past_transcriptions` | List all past transcriptions (in-memory) |
| `GET` | `/past_transcriptions/<id>` | Get a specific past transcription |

Full API reference → [`docs/API.md`](docs/API.md)

Deployment guide → [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)

---

## Monitoring & Dashboard

A **Streamlit dashboard** is deployed on Cloud Run and provides a public interface to the system:

```
https://hearedit-dashboard-726024632692.europe-west1.run.app
```

It has two pages:

| Page | Description |
|------|-------------|
| **Transcrire** | Upload an audio file, call the Flask API in real time, visualise segments on an interactive Gantt chart and speaker donut chart |
| **Historique** | Browse all past transcriptions stored in BigQuery — daily bar chart + expandable transcript list |

Every successful `/transcribe` API call automatically writes one row to **BigQuery** (`hearedit_dataset.transcriptions`), providing a persistent audit log and usage metrics without any client-side instrumentation.

```
User (browser)
    │
    ▼
Streamlit Dashboard (Cloud Run)          src/streamlit/dashboard.py
    │
    ├── POST /transcribe ──► Flask API (Cloud Run)
    │                            └── HearEdit pipeline (Vosk + ECAPA)
    │
    └── SELECT ──────────► BigQuery
                               └── hearedit_dataset.transcriptions
```

### Run the dashboard locally

```bash
cd src/streamlit
gcloud auth application-default login   # BigQuery access
uv run streamlit run dashboard.py
# → http://localhost:8501
```

Full monitoring & dashboard documentation → [`docs/MONITORING.md`](docs/MONITORING.md)

---

## CI/CD

The pipeline is defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml). CI runs on every pull request targeting `main` or `develop`.

Tests are intentionally lightweight: they require no models or audio files and run in seconds on any CI machine.

| Jobs | Tool | What it checks |
|------|------|----------------|
| Pre-commit | `pre-commit` + `ruff` | Code formatting, trailing whitespace, import sorting |
| Unit tests | `pytest` | Pure ML logic — normalize, IQR filter, EER, cosine similarity |

CD deploys the image in google cloud when pushing to `main`.

| Jobs | Tool | What it checks |
|------|------|----------------|
| Deploy | `gcloud` | Authenticate, build and push Docker image. Run deploy |

---

## Documentation

| File | Content |
|------|---------|
| [`docs/EXPERIMENTATION.md`](docs/EXPERIMENTATION.md) | Threshold calibration methodology, similarity distributions, ROC curve, ECAPA fine-tuning results |
| [`docs/DATA.md`](docs/DATA.md) | Cloud storage design (GCS vs BigQuery vs Firestore), bucket layout, upload/download process |
| [`docs/API.md`](docs/API.md) | Full REST API reference — endpoints, request/response format, implementation details |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Docker packaging, Cloud Run deployment, environment variables, memory considerations |
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | Vertex AI KFP pipeline — DAG, component details, Docker image, output artifacts |
| [`docs/MONITORING.md`](docs/MONITORING.md) | Streamlit dashboard architecture, BigQuery schema, deployment & local dev guide |
| [`docs/SPEAKER_ENCODER.md`](docs/SPEAKER_ENCODER.md) | ECAPA-TDNN speaker encoder — architecture, training procedure, fine-tuning details |
