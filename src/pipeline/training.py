from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from config import BASE_IMAGE

@component(base_image=BASE_IMAGE)
def training(
    train_split: Input[Dataset],
    val_split: Input[Dataset],
    gcs_dataset_uri: str,
    model: Output[Model],
    metrics: Output[Metrics],
    epochs: int = 20,
    batch_size: int = 16,
):
    import os, json, random
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torchaudio
    import soundfile as sf
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from google.cloud import storage

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **k: None

    from speechbrain.inference.speaker import EncoderClassifier

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    LOCAL_MODEL_DIR = "/tmp/ecapa_model"

    # --- Télécharger les données depuis GCS ---
    def download_gcs_prefix(gcs_uri, local_dir):
        uri = gcs_uri.replace("gs://", "")
        bucket_name, prefix = uri.split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            rel_path = blob.name[len(prefix):]
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if not blob.name.endswith("/"):
                blob.download_to_filename(local_path)
        print(f"Downloaded {gcs_uri} -> {local_dir}")

    download_gcs_prefix(gcs_dataset_uri, "/tmp/librispeech")

    # --- Charger les splits ---
    with open(os.path.join(train_split.path, "train.json")) as f:
        train_items = json.load(f)
    with open(os.path.join(val_split.path, "val.json")) as f:
        val_items = json.load(f)

    print(f"Train: {len(train_items)} utterances")
    print(f"Val:   {len(val_items)} utterances")

    # --- Construire le mapping speaker ---
    spk_to_idx = {s: i for i, s in enumerate(sorted({x["speaker"] for x in train_items}))}
    val_items = [x for x in val_items if x["speaker"] in spk_to_idx]

    print(f"Speakers: {len(spk_to_idx)}")

    # --- ArcFace Loss ---
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

    # --- Utilitaires ---
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

    # --- Dataset ---
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
        padded = [torch.nn.functional.pad(x, (0, max_len - x.shape[0])) for x in signals]
        rel_lengths = torch.tensor([l / max_len for l in lengths])
        return torch.stack(padded), rel_lengths, torch.stack(labels)

    train_loader = DataLoader(SpeakerDataset(train_items, spk_to_idx),
                              batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(SpeakerDataset(val_items, spk_to_idx),
                              batch_size=batch_size, collate_fn=collate_fn)

    # --- Charger encodeur ---
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=LOCAL_MODEL_DIR,
    )
    encoder.mods = encoder.mods.to(DEVICE)

    # Détecter dimension embedding
    tmp = load_audio_mono_16k(train_items[0]["wav"]).to(DEVICE)
    tmp_len = torch.tensor([1.0], device=DEVICE)
    with torch.no_grad():
        feats = encoder.mods.compute_features(tmp)
        feats = encoder.mods.mean_var_norm(feats, tmp_len)
        emb = encoder.mods.embedding_model(feats, tmp_len)
    emb_dim = emb.squeeze().shape[-1]
    print(f"Embedding dim: {emb_dim}")

    # --- ArcFace + optimizer ---
    criterion = ArcFaceLoss(emb_dim, len(spk_to_idx)).to(DEVICE)
    optimizer = optim.Adam([
        {"params": encoder.mods.embedding_model.parameters(), "lr": 3e-6},
        {"params": criterion.parameters(), "lr": 5e-4},
    ])

    # --- Entraînement ---
    print(f"Début de l'entraînement sur {DEVICE}")
    print(f"Nb batches train: {len(train_loader)}")
    best_val_acc = 0.0
    for epoch in range(epochs):
        print(f"Début epoch {epoch+1}/{epochs}")
        encoder.mods.embedding_model.train()
        criterion.train()
        total_loss = 0.0
        for i, (signals, lengths, labels) in enumerate(train_loader):
            if i == 0:
                print(f"Premier batch OK — shape: {signals.shape}")
            signals, lengths, labels = signals.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            feats = encoder.mods.compute_features(signals)
            feats = encoder.mods.mean_var_norm(feats, lengths)
            emb = encoder.mods.embedding_model(feats, lengths).squeeze(1)
            loss = criterion(emb, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        encoder.mods.embedding_model.eval()
        criterion.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for signals, lengths, labels in val_loader:
                signals, lengths, labels = signals.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
                feats = encoder.mods.compute_features(signals)
                feats = encoder.mods.mean_var_norm(feats, lengths)
                emb = encoder.mods.embedding_model(feats, lengths).squeeze(1)
                emb_norm = F.normalize(emb, dim=1)
                W_norm = F.normalize(criterion.weight, dim=1)
                cos_sim = torch.mm(emb_norm, W_norm.t())
                preds = cos_sim.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / max(1, total)
        print(f"Epoch {epoch+1}/{epochs} — train_loss={avg_loss:.4f} val_acc={val_acc:.4f}")
        metrics.log_metric(f"train_loss_epoch_{epoch+1}", avg_loss)
        metrics.log_metric(f"val_acc_epoch_{epoch+1}", val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(model.path, exist_ok=True)
            torch.save(encoder.mods.embedding_model.state_dict(),
                       os.path.join(model.path, "encoder.pt"))
            torch.save({"spk_to_idx": spk_to_idx},
                       os.path.join(model.path, "speaker_labels.pt"))
            print(f"  → Meilleurs poids sauvegardés (val_acc={val_acc:.4f})")

    metrics.log_metric("best_val_acc", best_val_acc)
    metrics.log_metric("num_speakers", len(spk_to_idx))
    print(f"Training done. best_val_acc={best_val_acc:.4f}")