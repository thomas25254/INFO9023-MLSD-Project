# Data Storage — Design Decisions

## Overview

This document describes how the project data is stored in the cloud and justifies the choice of storage service.

---

## Cloud Storage Choice: Google Cloud Storage (Blob Storage)

We selected **Google Cloud Storage (GCS)** as our cloud data storage solution.

### Options considered

| Service | Type | Best suited for |
|---|---|---|
| **BigQuery** | Relational / analytical SQL | Structured tabular data, SQL queries |
| **Firestore** | NoSQL document store | Real-time apps, document-based data |
| **Cloud Storage** | Blob / object storage | Large binary files, unstructured data |

### Why GCS?

Our project handles exclusively **unstructured data**:

- Raw audio recordings (`.flac` files, up to several MB each)
- Transcription outputs (`.txt` files)
- Trained speaker embedding caches (`.npz` binary arrays)
- Model artefacts

Object storage is the natural fit for this type of data. BigQuery requires structured schemas and is optimized for analytical SQL queries — irrelevant for audio files. Firestore is designed for document-oriented real-time applications, not large binary file storage.

GCS provides:

- **Scalability**: handles large datasets without schema management
- **Cost efficiency**: pay-per-use, no idle compute cost
- **Native GCP integration**: works directly with the Python `google-cloud-storage` SDK and with GCP IAM for access control

---

## Bucket Configuration

| Parameter | Value |
|---|---|
| **Project** | `info9023-project-hearedit` |
| **Bucket name** | `mlops-2026-dataset-bucket` |
| **Location** | `europe-west1` (Belgium) |
| **Storage class** | Standard |


---

## Data Structure in GCS

```
gs://mlops-2026-dataset-bucket/
└── dev-clean/
    └── LibriSpeech/
        ├── SPEAKERS.TXT
        ├── CHAPTERS.TXT
        └── dev-clean/
            └── <speaker_id>/
                └── <chapter_id>/
                    ├── <speaker_id>-<chapter_id>.trans.txt
                    └── *.flac
```

This mirrors the original LibriSpeech folder structure exactly, so the training script works without any path remapping whether data is local or downloaded from GCS.

---

## Upload Process

Data was uploaded once using `src/gcs_upload.ipynb`. The notebook uses `google-cloud-storage` to recursively walk the local LibriSpeech directory and upload each file, preserving the folder hierarchy:

```python
upload_folder_to_gcs(
    project_id="info9023-project-hearedit",
    bucket_name="mlops-2026-dataset-bucket",
    source_folder="LibriSpeech",
    destination_folder="dev-clean/LibriSpeech"
)
```

This upload only needs to be run **once**. All subsequent training runs download data from GCS automatically if not already present locally.

---

## Download Process at Training Time

The training script (`speaker_embedding_threshold.py`) automatically syncs data from GCS if the local dataset directory is missing or incomplete:

```python
# Triggered automatically when --local flag is NOT used
sync_gcs_prefix_to_dir(
    bucket_name="mlops-2026-dataset-bucket",
    prefix="dev-clean/LibriSpeech/",
    dest_dir="./data/LibriSpeech/",
    skip_if_exists=True,   # avoids re-downloading existing files
)
```

The `skip_if_exists=True` flag ensures already-downloaded files are never re-fetched, making repeated runs efficient.

---

## Authentication

| Environment | Method |
|---|---|
| **Local development** | `gcloud auth application-default login` |
| **Docker / Cloud Run** | `GOOGLE_APPLICATION_CREDENTIALS` env var pointing to a service account key |
| **GCP Compute (future)** | Workload Identity / instance service account |

No credentials are ever committed to the repository. The `.gitignore` excludes all `*.json` key files.

---

## What is NOT stored on GCS

| Item | Reason |
|---|---|
| Vosk ASR model (`vosk-model-en-us-0.22`) | Downloaded directly from https://alphacephei.com/vosk/models |
| Vosk speaker model (`vosk-model-spk-0.4`) | Downloaded directly from https://alphacephei.com/vosk/models |
| Embedding cache (`.npz`) | Local artefact, regenerated from data |
| Threshold JSON | Local artefact, output of the training script |

Model weights are not versioned in GCS at this stage. This will be revisited when model versioning becomes a requirement (Sprint 3+).
