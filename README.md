# HearEdit 🎙️

> **INFO9023 — Machine Learning Systems Design**

HearEdit is an automatic **speaker diarization and transcription** pipeline. Given an audio recording of a conversation, it identifies *who spoke when*, transcribes the speech, and assigns each segment to a speaker — even speakers never seen during training.

---

## Table of Contents

- [Use Case](#use-case)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Data](#data)
- [ML Model](#ml-model)
- [CI/CD](#cicd)
- [Documentation](#documentation)
- [Milestones](#milestones)

---

## Use Case

Many real-world recordings (meetings, interviews, podcasts, lectures) involve multiple speakers. HearEdit automatically segments and labels those recordings by speaker identity, producing a structured transcript. The core ML component is a **speaker verification model** based on cosine similarity of voice embeddings extracted by [Vosk](https://alphacephei.com/vosk/).

At inference time, each new audio segment is compared to known speaker prototypes. If the cosine similarity exceeds a learned threshold, it is assigned to that speaker; otherwise a new speaker is created. The threshold is determined empirically on the LibriSpeech `dev-clean` dataset using the **Equal Error Rate (EER)** criterion.

---

## Project Structure

```
hearedit/
│
├── .github/
│   └── workflows/
│       └── ci.yml                  # CI/CD pipeline (pre-commit + pytest)
│
├── src/
│   └── milestone_1/
│       ├── __init__.py
│       ├── utils.py                        # Pure, testable functions (normalize, IQR filter, EER…)
│       ├── gcs_download.py                 # GCS → local data sync utility
│       ├── speaker_embedding_threshold.py  # Main training/threshold script
│       ├── vosk_speaker_recognition.ipynb  # EDA & prototyping notebook
|       └── data/                           # gitignored — local LibriSpeech copy
│          └── LibriSpeech/
│              ├── SPEAKERS.TXT
│              ├── CHAPTERS.TXT
│              └── dev-clean/
│
├── docs/
│   ├── EXPERIMENTATION.md          # EDA results, similarity analysis, threshold selection
│   └── DATA.md                     # Cloud storage design decisions
│
├── tests/
│   └── test_speaker_embedding.py   # Unit tests (normalize, IQR, EER, similarities)
│
├── slides/
│   └── milestone_1.pdf             # Milestone 1 presentation slides
│
├── data/                           # gitignored — local LibriSpeech copy
│   └── LibriSpeech/
│       ├── SPEAKERS.TXT
│       ├── CHAPTERS.TXT
│       └── dev-clean/
│
├── models/
│   ├── vosk-model-en-us-0.22/
│   └── vosk-model-spk-0.4/
│
├── .pre-commit-config.yaml         # Pre-commit hooks (ruff formatting)
├── .gitignore
├── pyproject.toml                  # Dependencies managed with uv
└── README.md
```

---

## Getting Started


### Download Vosk models

Download and extract the following models into `./models/`:

| Model | Link |
|---|---|
| `vosk-model-en-us-0.22` | https://alphacephei.com/vosk/models |
| `vosk-model-spk-0.4` | https://alphacephei.com/vosk/models |

```
models/
├── vosk-model-en-us-0.22/
└── vosk-model-spk-0.4/
```

### Run the threshold computation script

**Option A — Local data** (LibriSpeech already downloaded):

```bash
python src/speaker_embedding_threshold.py \
    --local \
    --ds_dir    ./data/LibriSpeech/ \
    --vosk_model ./models/vosk-model-en-us-0.22 \
    --spk_model  ./models/vosk-model-spk-0.4
```

**Option B — Auto-download from GCS** (requires GCP credentials):

```bash
gcloud auth application-default login

python src/speaker_embedding_threshold.py \
    --ds_dir    ./data/LibriSpeech/ \
    --vosk_model ./models/vosk-model-en-us-0.22 \
    --spk_model  ./models/vosk-model-spk-0.4
```

The script produces:
- `speakers_embeddings_dev-clean.npz` — cached embeddings (reused on subsequent runs)
- `threshold_dev-clean.json` — optimal cosine similarity threshold + metrics

### Run the tests

```bash
PYTHONPATH=src pytest tests/ -v
```

---

## Data

The dataset used is **LibriSpeech `dev-clean`** — 40 speakers, read English audiobooks in `.flac` format.

Data is stored on **Google Cloud Storage**:

```
gs://mlops-2026-dataset-bucket/dev-clean/LibriSpeech/
```

The choice of GCS over BigQuery or Firestore is documented in [`docs/DATA.md`](docs/DATA.md).

> `data/` and `models/` are listed in `.gitignore` and are never committed to the repository.

---

## ML Model

The speaker verification model is based on **cosine similarity** of voice embeddings extracted by the Vosk speaker model (`vosk-model-spk-0.4`).

The core steps are:

1. Extract embeddings for all speakers in LibriSpeech `dev-clean`
2. Filter outlier speakers using IQR
3. Compute pairwise intra-speaker and inter-speaker cosine similarities
4. Fit a ROC curve and select the threshold at the **Equal Error Rate (EER)**

Full methodology and results are in [`docs/EXPERIMENTATION.md`](docs/EXPERIMENTATION.md).

---

## CI/CD

The pipeline is defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and runs on every pull request targeting `main` or `develop`.

| Step | Tool | What it checks |
|---|---|---|
| Pre-commit hooks | `pre-commit` | Code formatting (ruff), trailing whitespace, etc. |
| Unit tests | `pytest` | Pure ML logic — normalize, IQR filter, EER threshold, similarities |

The tests are intentionally **lightweight**: they do not require Vosk models or audio files, so they run in seconds on any CI machine.

---

## Documentation

| File | Content |
|---|---|
| [`docs/EXPERIMENTATION.md`](docs/EXPERIMENTATION.md) | EDA, similarity distributions, threshold selection methodology and results |
| [`docs/DATA.md`](docs/DATA.md) | Cloud storage choice (GCS vs BigQuery vs Firestore), bucket config, upload & download process |

---
