from kfp.dsl import component, Input, Output, Model, Metrics
from config import BASE_IMAGE

@component(base_image=BASE_IMAGE)
def compute_threshold(
    gcs_dataset_uri: str,
    gcs_threshold_output_uri: str,
    model: Input[Model],
    metrics: Output[Metrics],
):
    import json, os
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchaudio
    import soundfile as sf
    from collections import defaultdict
    from sklearn.metrics import auc, roc_curve
    from sklearn.metrics.pairwise import cosine_similarity
    from google.cloud import storage

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **k: None

    from speechbrain.inference.speaker import EncoderClassifier

    DEVICE = "cpu"
    LOCAL_DATA = "/tmp/librispeech"
    LOCAL_MODEL_DIR = "/tmp/ecapa_model"

    # --- Télécharger les données ---
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

    download_gcs_prefix(gcs_dataset_uri, LOCAL_DATA)

    # --- Charger notre modèle ECAPA fine-tuné ---
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=LOCAL_MODEL_DIR,
    )
    encoder.mods.embedding_model.load_state_dict(
        torch.load(os.path.join(model.path, "encoder.pt"), map_location=DEVICE)
    )
    encoder.mods = encoder.mods.to(DEVICE)
    encoder.mods.embedding_model.eval()

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

    def extract_embedding(path):
        signal = load_audio_mono_16k(path).to(DEVICE)
        lengths = torch.tensor([1.0], device=DEVICE)
        with torch.no_grad():
            feats = encoder.mods.compute_features(signal)
            feats = encoder.mods.mean_var_norm(feats, lengths)
            emb = encoder.mods.embedding_model(feats, lengths)
        return F.normalize(emb.squeeze(), dim=0).cpu().numpy()

    # --- Extraire embeddings par speaker ---
    by_spk = defaultdict(list)
    for root, _, files in os.walk(LOCAL_DATA):
        for f in files:
            if f.endswith(".flac"):
                parts = os.path.join(root, f).replace("\\", "/").split("/")
                speaker_id = parts[-3]
                by_spk[speaker_id].append(os.path.join(root, f))

    speakers_embeddings = {}
    for spk, paths in by_spk.items():
        embs = [extract_embedding(p) for p in paths]
        if embs:
            speakers_embeddings[spk] = np.array(embs)

    print(f"Threshold computation sur {len(speakers_embeddings)} speakers")

    # --- Calculer EER ---
    def normalize(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    filtered = [normalize(e) for e in speakers_embeddings.values() if len(e) >= 2]

    intra, inter = [], []
    for i, e in enumerate(filtered):
        sim = cosine_similarity(e)
        n = sim.shape[0]
        intra.extend(sim[np.triu_indices(n, k=1)])
        for j in range(i+1, len(filtered)):
            inter.extend(cosine_similarity(e, filtered[j]).ravel())

    scores = np.concatenate([np.array(intra), np.array(inter)])
    labels = np.concatenate([np.ones(len(intra)), np.zeros(len(inter))])

    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    roc_auc = float(auc(fpr, tpr))
    eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
    eer = float((fpr[eer_idx] + (1 - tpr[eer_idx])) / 2)
    threshold = float(thresholds[eer_idx])

    print(f"EER={eer:.4f}, AUC={roc_auc:.4f}, Threshold={threshold:.4f}")

    # --- Sauvegarder ---
    payload = {
        "threshold": threshold,
        "eer": eer,
        "auc": roc_auc,
    }
    local_json = "/tmp/threshold.json"
    with open(local_json, "w") as f:
        json.dump(payload, f, indent=2)

    uri = gcs_threshold_output_uri.replace("gs://", "")
    bucket_name, blob_path = uri.split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_path).upload_from_filename(local_json)
    print(f"Threshold saved to {gcs_threshold_output_uri}")

    metrics.log_metric("eer", eer)
    metrics.log_metric("auc", roc_auc)
    metrics.log_metric("threshold", threshold)