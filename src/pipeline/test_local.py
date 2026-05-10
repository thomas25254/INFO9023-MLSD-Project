import glob
import json
import os
import random
from collections import defaultdict


# ============================================================
# TEST DATA PREPARATION
# ============================================================
def test_data_preparation():
    print("=== TEST DATA PREPARATION ===")

    LOCAL_DATA = (
        "/home/alexandre/Documents/2025-2026/Q2/MLSD/Project/LibriSpeech/dev-clean"
    )
    OUTPUT_DIR = "/tmp/test_pipeline"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def discover_files(root):
        files = glob.glob(os.path.join(root, "**", "*.flac"), recursive=True)
        items = []
        for p in sorted(files):
            parts = p.replace("\\", "/").split("/")
            speaker_id = parts[-3]
            items.append({"wav": p, "speaker": speaker_id})
        return items

    all_items = discover_files(LOCAL_DATA)
    print(f"Fichiers trouvés: {len(all_items)}")

    by_spk = defaultdict(list)
    for item in all_items:
        by_spk[item["speaker"]].append(item)

    train_items, val_items, test_items = [], [], []
    for _spk, utts in by_spk.items():
        if len(utts) < 3:
            continue
        random.shuffle(utts)
        n_train = max(1, int(len(utts) * 0.8))
        n_val = max(1, int(len(utts) * 0.1))
        train_items.extend(utts[:n_train])
        val_items.extend(utts[n_train : n_train + n_val])
        test_items.extend(utts[n_train + n_val :])

    print(f"Train: {len(train_items)}, Val: {len(val_items)}, Test: {len(test_items)}")

    with open(f"{OUTPUT_DIR}/train.json", "w") as f:
        json.dump(train_items, f)
    with open(f"{OUTPUT_DIR}/val.json", "w") as f:
        json.dump(val_items, f)
    with open(f"{OUTPUT_DIR}/test.json", "w") as f:
        json.dump(test_items, f)

    print("Data preparation OK ✅")
    return OUTPUT_DIR


# ============================================================
# TEST TRAINING (2 epochs seulement pour tester)
# ============================================================
def test_training(split_dir):
    print("\n=== TEST TRAINING ===")

    import soundfile as sf
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset as TorchDataset

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **k: None

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    with open(f"{split_dir}/train.json") as f:
        train_items = json.load(f)
    with open(f"{split_dir}/val.json") as f:
        val_items = json.load(f)

    # Prendre juste 50 utterances pour tester rapidement
    train_items = train_items[:50]
    val_items = val_items[:10]

    spk_to_idx = {
        s: i for i, s in enumerate(sorted({x["speaker"] for x in train_items}))
    }
    val_items = [x for x in val_items if x["speaker"] in spk_to_idx]

    print(
        f"Train: {len(train_items)}, Val: {len(val_items)}, Speakers: {len(spk_to_idx)}"
    )

    class ArcFaceLoss(nn.Module):
        def __init__(self, emb_dim, num_speakers, s=30.0, m=0.5):
            super().__init__()
            self.s = s
            self.m = m
            self.weight = nn.Parameter(torch.FloatTensor(num_speakers, emb_dim))
            nn.init.xavier_uniform_(self.weight)

        def forward(self, emb, labels):
            emb = F.normalize(emb, dim=1)
            W = F.normalize(self.weight, dim=1)
            cos_theta = torch.mm(emb, W.t()).clamp(-1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos_theta)
            theta_m = theta + self.m
            cos_theta_m = torch.cos(theta_m)
            one_hot = torch.zeros_like(cos_theta)
            one_hot.scatter_(1, labels.view(-1, 1), 1)
            logits = one_hot * cos_theta_m + (1 - one_hot) * cos_theta
            logits *= self.s
            return F.cross_entropy(logits, labels)

    def load_audio_mono_16k(path, max_seconds=6.0):
        signal, sr = sf.read(path, dtype="float32")
        if len(signal.shape) == 2:
            signal = signal.mean(axis=1)
        signal = torch.tensor(signal).unsqueeze(0)
        if sr != 16000:
            signal = torchaudio.transforms.Resample(sr, 16000)(signal)
        max_len = int(max_seconds * 16000)
        if signal.shape[1] > max_len:
            signal = signal[:, :max_len]
        return signal

    class SpeakerDataset(TorchDataset):
        def __init__(self, items, spk_to_idx):
            self.items = items
            self.spk_to_idx = spk_to_idx

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            item = self.items[idx]
            signal = load_audio_mono_16k(item["wav"]).squeeze(0)
            label = torch.tensor(self.spk_to_idx[item["speaker"]], dtype=torch.long)
            return signal, label

    def collate_fn(batch):
        signals, labels = zip(*batch, strict=False)
        lengths = [x.shape[0] for x in signals]
        max_len = max(lengths)
        padded = [
            torch.nn.functional.pad(x, (0, max_len - x.shape[0])) for x in signals
        ]
        rel_lengths = torch.tensor([length / max_len for length in lengths])
        return torch.stack(padded), rel_lengths, torch.stack(labels)

    train_loader = DataLoader(
        SpeakerDataset(train_items, spk_to_idx),
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
    )

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/tmp/ecapa_model",
    )
    encoder.mods = encoder.mods.to(DEVICE)

    tmp = load_audio_mono_16k(train_items[0]["wav"]).to(DEVICE)
    tmp_len = torch.tensor([1.0], device=DEVICE)
    with torch.no_grad():
        feats = encoder.mods.compute_features(tmp)
        feats = encoder.mods.mean_var_norm(feats, tmp_len)
        emb = encoder.mods.embedding_model(feats, tmp_len)
    emb_dim = emb.squeeze().shape[-1]
    print(f"Embedding dim: {emb_dim}")

    criterion = ArcFaceLoss(emb_dim, len(spk_to_idx)).to(DEVICE)
    optimizer = optim.Adam(
        [
            {"params": encoder.mods.embedding_model.parameters(), "lr": 3e-6},
            {"params": criterion.parameters(), "lr": 5e-4},
        ]
    )

    # 2 epochs seulement pour le test
    for epoch in range(2):
        encoder.mods.embedding_model.train()
        criterion.train()
        total_loss = 0.0
        for signals, lengths, labels in train_loader:
            signals, lengths, labels = (
                signals.to(DEVICE),
                lengths.to(DEVICE),
                labels.to(DEVICE),
            )
            optimizer.zero_grad()
            feats = encoder.mods.compute_features(signals)
            feats = encoder.mods.mean_var_norm(feats, lengths)
            emb = encoder.mods.embedding_model(feats, lengths).squeeze(1)
            loss = criterion(emb, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/2 — train_loss={total_loss / len(train_loader):.4f}")

    # Sauvegarder
    MODEL_DIR = "/tmp/test_pipeline/model"
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(encoder.mods.embedding_model.state_dict(), f"{MODEL_DIR}/encoder.pt")
    torch.save({"spk_to_idx": spk_to_idx}, f"{MODEL_DIR}/speaker_labels.pt")

    print("Training OK ✅")
    return MODEL_DIR


# ============================================================
# TEST EVALUATION
# ============================================================
def test_evaluation(split_dir, model_dir):
    print("\n=== TEST EVALUATION ===")

    from collections import defaultdict

    import numpy as np
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    import torchaudio
    from sklearn.metrics import auc, roc_curve
    from sklearn.metrics.pairwise import cosine_similarity
    from speechbrain.inference.speaker import EncoderClassifier

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **k: None

    DEVICE = "cpu"

    with open(f"{split_dir}/test.json") as f:
        test_items = json.load(f)
    print(f"Test: {len(test_items)} utterances")

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/tmp/ecapa_model",
    )
    encoder.mods.embedding_model.load_state_dict(
        torch.load(f"{model_dir}/encoder.pt", map_location=DEVICE)
    )
    encoder.mods = encoder.mods.to(DEVICE)
    encoder.mods.embedding_model.eval()

    def load_audio_mono_16k(path, max_seconds=6.0):
        signal, sr = sf.read(path, dtype="float32")
        if len(signal.shape) == 2:
            signal = signal.mean(axis=1)
        signal = torch.tensor(signal).unsqueeze(0)
        if sr != 16000:
            signal = torchaudio.transforms.Resample(sr, 16000)(signal)
        max_len = int(max_seconds * 16000)
        if signal.shape[1] > max_len:
            signal = signal[:, :max_len]
        return signal

    def extract_embedding(path):
        signal = load_audio_mono_16k(path).to(DEVICE)
        lengths = torch.tensor([1.0], device=DEVICE)
        with torch.no_grad():
            feats = encoder.mods.compute_features(signal)
            feats = encoder.mods.mean_var_norm(feats, lengths)
            emb = encoder.mods.embedding_model(feats, lengths)
        return F.normalize(emb.squeeze(), dim=0).cpu().numpy()

    by_spk = defaultdict(list)
    for item in test_items:
        by_spk[item["speaker"]].append(item["wav"])

    speakers_embeddings = {}
    for spk, paths in by_spk.items():
        embs = [extract_embedding(p) for p in paths]
        if embs:
            speakers_embeddings[spk] = np.array(embs)

    intra, inter = [], []
    spk_list = list(speakers_embeddings.values())
    for i, e in enumerate(spk_list):
        if len(e) < 2:
            continue
        sim = cosine_similarity(e)
        n = sim.shape[0]
        intra.extend(sim[np.triu_indices(n, k=1)])
        for j in range(i + 1, len(spk_list)):
            inter.extend(cosine_similarity(e, spk_list[j]).ravel())

    scores = np.concatenate([np.array(intra), np.array(inter)])
    labels = np.concatenate([np.ones(len(intra)), np.zeros(len(inter))])

    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    roc_auc = float(auc(fpr, tpr))
    eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
    eer = float((fpr[eer_idx] + (1 - tpr[eer_idx])) / 2)
    threshold = float(thresholds[eer_idx])

    print(f"EER={eer:.4f}, AUC={roc_auc:.4f}, Threshold={threshold:.4f}")
    print("Evaluation OK")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    split_dir = test_data_preparation()
    model_dir = test_training(split_dir)
    test_evaluation(split_dir, model_dir)
    print("\n=== TOUS LES TESTS PASSÉS  ===")
