import os
import glob
import random
from collections import defaultdict

import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio

from tqdm import tqdm
import time
import csv

# -------------------------------------------------
# Compat torchaudio / speechbrain
# -------------------------------------------------
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda *args, **kwargs: None

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "models"
LIBRI_ROOT = "train-clean-100/LibriSpeech/train-clean-100"

ARTIFACT_DIR = "artifacts"
BEST_ENCODER_PATH = os.path.join(ARTIFACT_DIR, "ecapa_finetuned_speakerid_hidden512_Llr.pt")
BEST_CLASSIFIER_PATH = os.path.join(ARTIFACT_DIR, "speaker_classifier_head_hidden512_Llr.pt")
SPEAKER_LABELS_PATH = os.path.join(ARTIFACT_DIR, "speaker_label_mapping_hidden512_Llr.pt")

SEED = 42
BATCH_SIZE = 16
EPOCHS = 10
LR_ENCODER = 3e-6 #LR_ENCODER = 1e-5
LR_CLASSIFIER = 5e-4 #LR_CLASSIFIER = 1e-3
MAX_AUDIO_SECONDS = 6.0

TRAIN_UTT_RATIO = 0.8
VAL_UTT_RATIO = 0.2

CLASSIFIER_HIDDEN_DIM = 512


# =========================================================
# UTILS
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_audio_mono_16k(audio_path: str, max_seconds: float | None = None):
    signal, sr = sf.read(audio_path, dtype="float32")

    # stéréo -> mono
    if len(signal.shape) == 2:
        signal = signal.mean(axis=1)

    signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # [1, time]

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        signal = resampler(signal)
        sr = 16000

    if max_seconds is not None:
        max_len = int(max_seconds * sr)
        if signal.shape[1] > max_len:
            signal = signal[:, :max_len]

    return signal

def discover_librispeech_files(root_dir: str):
    flac_files = glob.glob(os.path.join(root_dir, "*", "*", "*.flac"))
    flac_files.sort()

    items = []
    for path in flac_files:
        parts = path.replace("\\", "/").split("/")
        if len(parts) < 3:
            continue
        speaker_id = parts[-3]
        items.append({"wav": path, "speaker": speaker_id})
    return items


# =========================================================
# LOAD SPEECHBRAIN ENCODER
# =========================================================
def load_sb_encoder(model_dir: str):
    classifier = EncoderClassifier.from_hparams(
        source=model_dir,
        savedir="pretrained_models/ecapa_local",
        local_strategy=LocalStrategy.COPY
    )
    classifier.device = DEVICE
    classifier.mods = classifier.mods.to(DEVICE)
    return classifier

def encode_audio_with_sb_trainable(classifier, signal_batch, lengths=None):
    """
    signal_batch: [B, T]
    lengths: [B] in [0,1]
    return: [B, D]
    """
    mods = classifier.mods

    feats = mods.compute_features(signal_batch)
    feats = mods.mean_var_norm(feats, lengths)
    emb = mods.embedding_model(feats, lengths)

    if emb.dim() == 3:
        emb = emb.squeeze(1)

    return emb


# =========================================================
# DATA PREPARATION
# =========================================================
def split_utterances_per_speaker(items, train_ratio=0.8, val_ratio=0.2, min_utts=2):
    by_speaker = defaultdict(list)
    for item in items:
        by_speaker[item["speaker"]].append(item)

    train_items = []
    val_items = []

    for spk, utts in by_speaker.items():
        if len(utts) < min_utts:
            continue

        random.shuffle(utts)
        n_train = max(1, int(len(utts) * train_ratio))
        n_train = min(n_train, len(utts) - 1)  # garder au moins 1 utt pour val si possible

        train_utts = utts[:n_train]
        val_utts = utts[n_train:]

        if len(val_utts) == 0:
            val_utts = [train_utts.pop()]

        train_items.extend(train_utts)
        val_items.extend(val_utts)

    return train_items, val_items

def build_speaker_label_map(items):
    speakers = sorted(list({x["speaker"] for x in items}))
    spk_to_idx = {spk: i for i, spk in enumerate(speakers)}
    idx_to_spk = {i: spk for spk, i in spk_to_idx.items()}
    return spk_to_idx, idx_to_spk


# =========================================================
# DATASET
# =========================================================
class SpeakerClassificationDataset(Dataset):
    def __init__(self, items, spk_to_idx, max_seconds=6.0):
        self.items = items
        self.spk_to_idx = spk_to_idx
        self.max_seconds = max_seconds

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        signal = load_audio_mono_16k(item["wav"], max_seconds=self.max_seconds).squeeze(0)
        label = torch.tensor(self.spk_to_idx[item["speaker"]], dtype=torch.long)
        return signal, label

def speaker_collate_fn(batch):
    signals, labels = zip(*batch)

    lengths = [x.shape[0] for x in signals]
    max_len = max(lengths)

    padded = []
    rel_lengths = []

    for x in signals:
        cur_len = x.shape[0]
        if cur_len < max_len:
            x = torch.nn.functional.pad(x, (0, max_len - cur_len))
        padded.append(x)
        rel_lengths.append(cur_len / max_len)

    padded = torch.stack(padded, dim=0)         # [B, T]
    rel_lengths = torch.tensor(rel_lengths, dtype=torch.float32)
    labels = torch.stack(labels, dim=0)

    return padded, rel_lengths, labels


# =========================================================
# CLASSIFIER HEAD
# =========================================================
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


# =========================================================
# TRAIN / EVAL
# =========================================================
def run_epoch_speakerid(sb_encoder, classifier_head, loader, criterion, optimizer=None, epoch_desc=""):
    is_train = optimizer is not None

    sb_encoder.mods.compute_features.eval()
    sb_encoder.mods.mean_var_norm.eval()

    if is_train:
        sb_encoder.mods.embedding_model.train()
        classifier_head.train()
    else:
        sb_encoder.mods.embedding_model.eval()
        classifier_head.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=epoch_desc, leave=True)

    for batch_idx, (signals, lengths, labels) in enumerate(pbar):
        signals = signals.to(DEVICE)
        lengths = lengths.to(DEVICE)
        labels = labels.to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            emb = encode_audio_with_sb_trainable(sb_encoder, signals, lengths)
            logits = classifier_head(emb)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        avg_loss = total_loss / (batch_idx + 1)
        acc = correct / max(1, total)

        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "acc": f"{acc:.4f}",
            "bs": labels.size(0)
        })

    avg_loss = total_loss / max(1, len(loader))
    acc = correct / max(1, total)

    return {"loss": avg_loss, "acc": acc}


# =========================================================
# TRAINING MAIN
# =========================================================
def train_speaker_id_finetune():
    print("torch:", torch.__version__)
    print("torchaudio:", torchaudio.__version__)
    print("DEVICE =", DEVICE)
    if torch.cuda.is_available():
        print("GPU =", torch.cuda.get_device_name(0))

    ensure_dir(ARTIFACT_DIR)
    ensure_dir("pretrained_models")
    set_seed(SEED)

    all_items = discover_librispeech_files(LIBRI_ROOT)
    if len(all_items) == 0:
        raise RuntimeError(f"Aucun .flac trouvé dans {LIBRI_ROOT}")

    print(f"Nombre total de fichiers audio trouvés: {len(all_items)}")

    train_items, val_items = split_utterances_per_speaker(
        all_items,
        train_ratio=TRAIN_UTT_RATIO,
        val_ratio=VAL_UTT_RATIO
    )

    # IMPORTANT:
    # le classifier speaker-ID apprend uniquement sur les speakers du train
    spk_to_idx, idx_to_spk = build_speaker_label_map(train_items)

    # on filtre val pour ne garder que les speakers connus du train
    train_speakers = set(spk_to_idx.keys())
    val_items = [x for x in val_items if x["speaker"] in train_speakers]

    print(f"Train utterances: {len(train_items)}")
    print(f"Val utterances:   {len(val_items)}")
    print(f"Nombre de speakers train: {len(spk_to_idx)}")

    train_dataset = SpeakerClassificationDataset(train_items, spk_to_idx, max_seconds=MAX_AUDIO_SECONDS)
    val_dataset = SpeakerClassificationDataset(val_items, spk_to_idx, max_seconds=MAX_AUDIO_SECONDS)

    print(f"Taille train_dataset: {len(train_dataset)}")
    print(f"Taille val_dataset:   {len(val_dataset)}")
    print(f"Batch size:          {BATCH_SIZE}")
    print(f"DEVICE utilisé:      {DEVICE}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=speaker_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=speaker_collate_fn
    )

    print(f"Nb batches train:    {len(train_loader)}")
    print(f"Nb batches val:      {len(val_loader)}")
    sb_encoder = load_sb_encoder(MODEL_DIR)

    # détecter dimension embedding
    tmp_signal = load_audio_mono_16k(train_items[0]["wav"], max_seconds=MAX_AUDIO_SECONDS).to(DEVICE)
    tmp_lengths = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        tmp_emb = encode_audio_with_sb_trainable(sb_encoder, tmp_signal, tmp_lengths)
    emb_dim = tmp_emb.shape[-1]

    print(f"Dimension embedding détectée: {emb_dim}")

    classifier_head = SpeakerClassifierHead(
        emb_dim=emb_dim,
        num_speakers=len(spk_to_idx),
        hidden_dim=CLASSIFIER_HIDDEN_DIM
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam([
        {"params": sb_encoder.mods.embedding_model.parameters(), "lr": LR_ENCODER},
        {"params": classifier_head.parameters(), "lr": LR_CLASSIFIER},
    ])

    best_val_acc = -1.0
    metrics_csv_path = os.path.join(ARTIFACT_DIR, "training_metrics_hidden512_Llr.csv")

    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "time_sec"])    

    for epoch in range(EPOCHS):
        t0 = time.time()

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

        dt = time.time() - t0

        print(
            f"\n[Epoch {epoch+1}/{EPOCHS}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"time={dt:.1f}s"
        )
        with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_metrics["loss"],
                train_metrics["acc"],
                val_metrics["loss"],
                val_metrics["acc"],
                dt
            ])

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(sb_encoder.mods.embedding_model.state_dict(), BEST_ENCODER_PATH)
            torch.save(classifier_head.state_dict(), BEST_CLASSIFIER_PATH)
            torch.save({
                "spk_to_idx": spk_to_idx,
                "idx_to_spk": idx_to_spk
            }, SPEAKER_LABELS_PATH)
            print("Meilleurs poids sauvegardés.")

    print("\nEntraînement terminé.")
    print(f"Encoder sauvegardé dans: {BEST_ENCODER_PATH}")
    print(f"Classifier head sauvegardé dans: {BEST_CLASSIFIER_PATH}")
    print(f"Mapping speakers sauvegardé dans: {SPEAKER_LABELS_PATH}")

    return sb_encoder, classifier_head, spk_to_idx, idx_to_spk


# =========================================================
# OPTIONAL: EXTRACTION D'EMBEDDING APRÈS FINE-TUNING
# =========================================================
def load_finetuned_encoder_only():
    sb_encoder = load_sb_encoder(MODEL_DIR)
    sb_encoder.mods.embedding_model.load_state_dict(
        torch.load(BEST_ENCODER_PATH, map_location=DEVICE)
    )
    sb_encoder.mods.embedding_model.eval()
    return sb_encoder

def extract_embedding_from_audio(sb_encoder, audio_path):
    signal = load_audio_mono_16k(audio_path, max_seconds=MAX_AUDIO_SECONDS).to(DEVICE)
    lengths = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        emb = encode_audio_with_sb_trainable(sb_encoder, signal, lengths)
    return emb.squeeze(0).cpu()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    train_speaker_id_finetune()