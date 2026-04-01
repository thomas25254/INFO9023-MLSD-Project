# Fine-tuning the speaker encoder

## What is this component used for?

In HearEdit, this component is used to convert an audio segment into a **speaker embedding**.
This embedding is then used by the diarizer to compare voices against one another.

It is important to distinguish between three key components in the project:

- **Vosk ASR**: a pre-trained model used to obtain the text and timestamps.
- **fine-tuned speaker encoder**: a model used to produce an embedding per segment.
- **diarizer**: a decision rule based on cosine similarity and an offline-calibrated threshold.

We have therefore **not** retrained Vosk.
The part that is actually trained in this component is the **speaker encoder**.

---


## Data used

We trained this component on:

- **Dataset**: `LibriSpeech train-clean-100`
- **Input unit**: an utterance associated with a `speaker_id`

The data is split into:

- **80% training**
- **20% validation**

The split is made at the level of utterances per speaker.

---

## Audio pre-processing

Before training, the audio files are:

- converted to **mono**
- resampled to **16 kHz**
- truncated to a **maximum of 6 seconds**


```python
def load_audio_mono_16k(audio_path: str, max_seconds: float | None = None):
    signal, sr = sf.read(audio_path, dtype="float32")

    if len(signal.shape) == 2:
        signal = signal.mean(axis=1)

    signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        signal = resampler(signal)
        sr = 16000

    if max_seconds is not None:
        max_len = int(max_seconds * sr)
        if signal.shape[1] > max_len:
            signal = signal[:, :max_len]

    return signal
```

This choice is consistent with the intended use of HearEdit, which involves processing short segments.

---

## Base model

The base model is loaded using SpeechBrain:

```python
def load_sb_encoder(model_dir: str):
    classifier = EncoderClassifier.from_hparams(
        source=model_dir,
        savedir="pretrained_models/ecapa_local",
        local_strategy=LocalStrategy.COPY
    )
    classifier.device = DEVICE
    classifier.mods = classifier.mods.to(DEVICE)
    return classifier
```

In practice, the part we’re interested in is the **embedding module**, This is the component that is fine-tuned and then reused later in HearEdit.

---

## How the training is formulated

We do not perform end-to-end diarization,we train the model as a **speaker identification** task.
To do this, we add a small classification head:

```python
class SpeakerClassifierHead(nn.Module):
    def __init__(self, emb_dim, num_speakers, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_speakers)
        )

    def forward(self, emb):
        return self.net(emb)
```

This head is only used during training.
It allows a supervised signal to be sent to the encoder.

---

## Selected configuration

Main script configuration:

| Parameter | Value |
|---|---:|
| Batch size | 16 |
| Epochs | 10 |
| Encoder learning rate | 1e-5 |
| Classification head learning rate | 5e-3 |
| Classifier head hidden dimension | 512 |
| Maximum segment length | 6.0 s |
| Loss | Cross-entropy |
| Optimizer | Adam |
| Best model selection | validation accuracy |

---

## Training loop

The logic is simple:

1. calculate the embeddings using the encoder.
2. apply the classification head.
3. calculate the loss.
4. train on the training set.
5. evaluate on the validation set.
6. save the model.


```python
train_metrics = run_epoch_speakerid(
    sb_encoder,
    classifier_head,
    train_loader,
    criterion,
    optimizer=optimizer,
    epoch_desc=f"Train {epoch+1}/{EPOCHS}"
)

val_metrics = run_epoch_speakerid(
    sb_encoder,
    classifier_head,
    val_loader,
    criterion,
    optimizer=None,
    epoch_desc=f"Val   {epoch+1}/{EPOCHS}"
)
```

---

## Variants tested

We compared several experiments:

- `training_metrics.csv`
- `training_metrics_hidden512.csv`
- `training_metrics_hidden512_b8.csv`
- `training_metrics_hidden512_Llr.csv`

These files correspond to different configuration variants.

---

## Results

| Experiment | Best epoch | Best val_acc | val_loss | Average time per epoch (s) |
|---|---:|---:|---:|---:|
| `training_metrics_hidden512.csv` | 6 | **0.998280** | 0.095592 | 249.34 |
| `training_metrics_hidden512_Llr.csv` | 5 | 0.997936 | **0.022553** | **246.16** |
| `training_metrics_hidden512_b8.csv` | 10 | 0.997592 | 0.160753 | 267.32 |
| `training_metrics.csv` | 5 | 0.997076 | 0.053376 | 258.45 |

### Quick read

- **best peak val_acc**: `training_metrics_hidden512.csv`
- **best val_loss**: `training_metrics_hidden512_Llr.csv`
- **best overall trade-off**: `training_metrics_hidden512_Llr.csv`

---

## Selected model

For this project, we selected the following configuration:

- **`training_metrics_hidden512.csv`**
- best epoch: **6**
- best `val_acc`: **0.998280**

We chose this variant because it produced the best results among the models tested, and it is the one that was ultimately retained in our pipeline.

Although other variants offered certain advantages on specific metrics, this configuration was used as the benchmark for the remainder of the project.

---

## Outputs

The script saves three elements:

- **the fine-tuned encoder**
- **the classification head**
- **the speaker ↔ label mapping**

However, in HearEdit, **only the fine-tuned encoder is reused**.

---

## Use in HearEdit

In the final pipeline:

1. **Vosk** generates the text and timestamps.
2. each audio segment is sent to the fine-tuned encoder.
3. the encoder produces a speaker embedding.
4. this embedding is fed to the diarizer.
5. the diarizer determines whether it is a known speaker or a new speaker.

The encoder is reloaded as follows:

```python
def load_finetuned_encoder_only(encoder_path):
    sb_encoder = load_sb_encoder(MODEL_DIR)
    sb_encoder.mods.embedding_model.load_state_dict(
        torch.load(encoder_path, map_location=DEVICE)
    )
    sb_encoder.mods.embedding_model.eval()
    return sb_encoder
```

Extracting an embedding:

```python
def extract_embedding_from_audio(sb_encoder, audio_path):
    signal = load_audio_mono_16k(audio_path, max_seconds=MAX_AUDIO_SECONDS).to(DEVICE)
    lengths = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        emb = encode_audio_with_sb_trainable(sb_encoder, signal, lengths)
    return emb.squeeze(0).cpu()
```

---

## In summary

The trained component of HearEdit is the **fine-tuned speaker encoder**.

We start with a pre-trained SpeechBrain model, which we fine-tune on a **speaker identification** task using `LibriSpeech train-clean-100`.

During training, we add a classification head.
At inference, we retain **only the encoder**, which is used to produce the embeddings subsequently used by the diarizer.

The **`hidden512`** variant is the one we have selected as the best compromise for the project.
