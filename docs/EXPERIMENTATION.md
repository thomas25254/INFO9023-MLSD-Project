# Experimentation — Speaker Recognition Threshold

## Overview

The goal of this experimentation is to determine an optimal **cosine similarity threshold** to discriminate between speakers in our audio diarization pipeline. The threshold is used at inference time to decide whether a new voice embedding belongs to a known speaker or corresponds to a new one.

---

## Method

### Model

We use **Vosk** (`vosk-model-spk-0.4`) as a voice feature extractor. For each audio segment, Vosk produces a fixed-size embedding vector that encodes the speaker's voice characteristics. These embeddings are then compared using **cosine similarity**.

### Dataset

- **LibriSpeech `dev-clean`** split
- 40 speakers, multiple chapters and audio samples per speaker
- Audio files in `.flac` format, converted to 16kHz mono PCM at runtime via `ffmpeg`

### Embedding strategy

For each speaker, all audio samples across all their chapters are processed. Vosk segments the audio internally and produces one embedding per segment. All segment embeddings are accumulated to form a per-speaker embedding matrix.

---

## Data Preparation

### IQR Filtering

To avoid bias from speakers with too few or too many samples, we apply an **IQR outlier filter** on the number of embeddings per speaker:

```
lower_bound = Q1 - 1.5 × IQR
upper_bound = Q3 + 1.5 × IQR
```

Speakers outside this range are excluded from the threshold computation. This ensures the intra/inter similarity distributions are not skewed by imbalanced speakers.

### Normalization

All embedding vectors are L2-normalized before computing cosine similarities, so that similarity scores are bounded in [-1, 1].

---

## Similarity Analysis

Two types of pairwise cosine similarities are computed:

- **Intra-speaker**: between embeddings from the **same** speaker
- **Inter-speaker**: between embeddings from **different** speakers

Similarities are computed **pair-wise on individual segments** (not on centroids), because at inference time a single new segment will be compared to known speaker prototypes.

| Metric | Intra-speaker | Inter-speaker |
|---|---|---|
| Mean | higher | lower |
| Distribution | right-skewed, centered near 1 | left-skewed, centered near 0 |

The two distributions overlap in the mid-range, which is exactly where the threshold needs to be placed.

---

## Threshold Selection

### Why EER?

Several threshold strategies were considered:

| Strategy | Description | Drawback |
|---|---|---|
| Mean / Median | Simple statistical cutoff | Ignores error trade-off |
| Minimize total error rate | Minimizes FA + FR | Biased towards majority class |
| **EER (Equal Error Rate)** | Balances False Acceptance and False Rejection | Best for symmetric speaker verification |

The **EER** is defined as the point on the ROC curve where:

```
False Positive Rate ≈ False Negative Rate
```

It represents the most balanced operating point for speaker verification, treating missed speaker matches and false speaker matches as equally undesirable.

### ROC Curve

The ROC curve is computed from the binary classification problem:
- **Positive label (1)**: intra-speaker pair (same speaker)
- **Negative label (0)**: inter-speaker pair (different speakers)

```
scores  = [intra_similarities..., inter_similarities...]
labels  = [1, 1, ..., 0, 0, ...]
```

### Results

| Metric | Value |
|---|---|
| **AUC** | 0.9823 |
| **EER** | 0.0950 |
| **Optimal threshold** | 0.4217 |

> **Note:** exact numeric values are saved in `threshold_dev-clean.json` after running `speaker_embedding_threshold.py`. The JSON also contains `n_speakers_total`, `n_speakers_after_iqr`, `n_intra_pairs`, and `n_inter_pairs` for full reproducibility.

---

## Artefacts

| File | Description |
|---|---|
| `speakers_embeddings_dev-clean.npz` | Cached speaker embeddings (reused across runs) |
| `threshold_dev-clean.json` | EER threshold + AUC + EER + metadata |
| `speaker_similarity_distributions.png` | Distribution plot of intra/inter similarities |

---

## How to Reproduce

```bash
# With data already local
python src/milestone_1/speaker_embedding_threshold.py \
    --local \
    --ds_dir ./data/LibriSpeech/ \
    --vosk_model ../models/vosk-model-en-us-0.22 \
    --spk_model  ../models/vosk-model-spk-0.4

# With data downloaded automatically from GCS
python src/milestone_1/speaker_embedding_threshold.py \
    --ds_dir ./data/LibriSpeech/ \
    --vosk_model ../models/vosk-model-en-us-0.22 \
    --spk_model  ../models/vosk-model-spk-0.4
```
