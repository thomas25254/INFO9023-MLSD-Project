# Sprint 5 — Dashboard & Monitoring

## Overview

This sprint implements a Streamlit dashboard deployed on Google Cloud Run. The dashboard provides a public web interface to interact with the HearEdit system: it allows users to transcribe audio files in real time via the Flask API, and to browse past transcriptions stored in BigQuery.

It is available in this URL : https://hearedit-dashboard-726024632692.europe-west1.run.app (we have scale to 0 and max to 2)

---

## Architecture

```
User (browser)
    │
    ▼
Streamlit Dashboard (Cloud Run)
    │
    ├── POST /transcribe ──► Flask API (Cloud Run)
    │                            └── HearEdit pipeline (Vosk + ECAPA)
    │
    └── SELECT ──────────► BigQuery
                               └── hearedit_dataset.transcriptions
```

**Serving mode:** Online (real-time). The dashboard calls the deployed Flask API for each transcription request. Results are automatically saved to BigQuery by the API and can be browsed in the History page.

---

## Components

### Streamlit App — `src/streamlit/dashboard.py`

The dashboard has two pages accessible via the sidebar:

**Transcrire**
- Audio file upload (WAV, MP3, OGG, FLAC)
- Calls `POST /transcribe` on the Flask API
- Displays:
  - Key metrics (segments, speakers, word count)
  - Full transcription text
  - Speaker breakdown — interactive donut chart (Plotly)
  - Timeline — interactive Gantt chart per speaker (Plotly)
  - Detailed segments with speaker badge and timestamps

**Historique**
- Reads past transcriptions from BigQuery (`hearedit_dataset.transcriptions`)
- Displays:
  - Summary metrics (total transcriptions, total segments, average segments)
  - Transcriptions per day — bar chart (Plotly)
  - Expandable list of all transcriptions with full text

### BigQuery Table — `hearedit_dataset.transcriptions`

The Flask API writes one row per transcription to this table after each successful `/transcribe` call.

| Column | Type | Description |
|--------|------|-------------|
| `filename` | STRING | Original audio filename |
| `full_text` | STRING | Complete transcription text |
| `num_segments` | INTEGER | Number of speaker segments |
| `created_at` | TIMESTAMP | UTC timestamp of the transcription |

### Dockerfile — `src/streamlit/Dockerfile`

```dockerfile
FROM mirror.gcr.io/library/python:3.13-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-hashes -o /tmp/requirements.txt && \
    uv pip install --system -r /tmp/requirements.txt
COPY dashboard.py .
ENV PORT=8501
CMD streamlit run dashboard.py \
    --server.port $PORT \
    --server.address=0.0.0.0 \
    --server.headless=true
```

---

## Cloud Infrastructure

| Resource | Value |
|----------|-------|
| Cloud Run service | `hearedit-dashboard` |
| Region | `europe-west1` |
| Port | `8501` |
| Service account | `726024632692-compute@developer.gserviceaccount.com` |
| BigQuery dataset | `info9023-project-hearedit.hearedit_dataset` |
| BigQuery table | `info9023-project-hearedit.hearedit_dataset.transcriptions` |

### IAM Roles granted to the service account

| Role | Purpose |
|------|---------|
| `roles/bigquery.dataEditor` | Write transcriptions from the Flask API |
| `roles/bigquery.jobUser` | Execute BigQuery queries from the dashboard |

---

## Deployment

### Prerequisites

```bash
# Enable required APIs (if not already done)
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  --project=info9023-project-hearedit

# Authenticate Docker
gcloud auth configure-docker europe-west1-docker.pkg.dev
```

### BigQuery setup (one-time)

```bash
bq mk --dataset --location=europe-west1 info9023-project-hearedit:hearedit_dataset

bq mk --table info9023-project-hearedit:hearedit_dataset.transcriptions \
  filename:STRING,full_text:STRING,num_segments:INTEGER,created_at:TIMESTAMP
```

### Deploy the dashboard

```bash
cd src/streamlit

gcloud run deploy hearedit-dashboard \
  --source . \
  --project=info9023-project-hearedit \
  --region=europe-west1 \
  --port=8501 \
  --set-env-vars="API_URL=https://hearedit-api-726024632692.europe-west1.run.app,BQ_TABLE=info9023-project-hearedit.hearedit_dataset.transcriptions" \
  --service-account=726024632692-compute@developer.gserviceaccount.com \
  --allow-unauthenticated
```

### Deploy the Flask API (with BigQuery integration)

The Flask API was updated to call `save_to_bigquery()` after each successful transcription. Redeploy with the `BQ_TABLE` environment variable:

```bash
gcloud run deploy hearedit-api \
  --project=info9023-project-hearedit \
  --image=europe-west1-docker.pkg.dev/info9023-project-hearedit/cloud-run-source-deploy/hearedit-api:latest \
  --region=europe-west1 \
  --set-env-vars="GCS_BUCKET=hearedit-models,BQ_TABLE=info9023-project-hearedit.hearedit_dataset.transcriptions,MODELS_DIR=/models,ARTIFACTS_DIR=/artifacts" \
  --min-instances=1 \
  --cpu=4 \
  --memory=16Gi \
  --timeout=600 \
  --allow-unauthenticated
```

---

## Local Development

```bash
cd src/streamlit

# Install dependencies
uv add streamlit requests pandas plotly google-cloud-bigquery db-dtypes

# Authenticate for local BigQuery access
gcloud auth application-default login

# Run the dashboard
uv run streamlit run dashboard.py
```

The dashboard will be available at `http://localhost:8501`.

The `API_URL` environment variable defaults to the deployed Cloud Run API. To point at a local Flask instance instead:

```bash
API_URL=http://localhost:5000 uv run streamlit run dashboard.py
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web dashboard framework |
| `plotly` | Interactive charts (donut, Gantt, bar) |
| `pandas` | Data manipulation |
| `requests` | HTTP calls to the Flask API |
| `google-cloud-bigquery` | Read/write BigQuery |
| `db-dtypes` | BigQuery type support for pandas |
