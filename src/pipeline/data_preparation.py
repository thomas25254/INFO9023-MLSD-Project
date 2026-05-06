from config import BASE_IMAGE
from kfp.dsl import Dataset, Output, component


@component(base_image=BASE_IMAGE)
def data_preparation(
    gcs_dataset_uri: str,
    train_split: Output[Dataset],
    val_split: Output[Dataset],
    test_split: Output[Dataset],
):
    import glob
    import json
    import os
    import random
    from collections import defaultdict

    from google.cloud import storage

    LOCAL_DATA = "/tmp/librispeech"

    # --- Download depuis GCS ---
    def download_gcs_prefix(gcs_uri, local_dir):
        uri = gcs_uri.replace("gs://", "")
        bucket_name, prefix = uri.split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            rel_path = blob.name[len(prefix) :]
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if not blob.name.endswith("/"):
                blob.download_to_filename(local_path)
        print(f"Downloaded {gcs_uri} -> {local_dir}")

    download_gcs_prefix(gcs_dataset_uri, LOCAL_DATA)

    # --- Découvrir les fichiers ---
    def discover_files(root):
        files = glob.glob(os.path.join(root, "**", "*.flac"), recursive=True)
        print(f"Structure exemple: {files[:3] if files else 'Aucun fichier'}")
        items = []
        for p in sorted(files):
            parts = p.replace("\\", "/").split("/")
            speaker_id = parts[-3]
            items.append({"wav": p, "speaker": speaker_id})
        return items

    all_items = discover_files(LOCAL_DATA)
    if not all_items:
        raise RuntimeError(f"Aucun .flac trouvé dans {LOCAL_DATA}")
    print(f"Fichiers trouvés: {len(all_items)}")

    # --- Split 80/10/10 par speaker ---
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

    print(f"Train: {len(train_items)} utterances")
    print(f"Val:   {len(val_items)} utterances")
    print(f"Test:  {len(test_items)} utterances")

    # --- Sauvegarder les splits ---
    os.makedirs(train_split.path, exist_ok=True)
    os.makedirs(val_split.path, exist_ok=True)
    os.makedirs(test_split.path, exist_ok=True)

    with open(os.path.join(train_split.path, "train.json"), "w") as f:
        json.dump(train_items, f)
    with open(os.path.join(val_split.path, "val.json"), "w") as f:
        json.dump(val_items, f)
    with open(os.path.join(test_split.path, "test.json"), "w") as f:
        json.dump(test_items, f)

    print("Splits sauvegardés.")
