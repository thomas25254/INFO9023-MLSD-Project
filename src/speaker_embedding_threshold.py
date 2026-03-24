import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from vosk import KaldiRecognizer, Model, SetLogLevel, SpkModel

from gcs_download import sync_gcs_prefix_to_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_dir",
        default="/opt/dataset/LibriSpeech/",
        help="LibriSpeech root containing SPEAKERS.TXT and CHAPTERS.TXT",
    )
    parser.add_argument(
        "--current_ds", default="dev-clean", help="Dataset split name (e.g. dev-clean)"
    )
    parser.add_argument(
        "--vosk_model",
        default="/opt/models/vosk-model-en-us-0.22",
        help="Vosk ASR model path",
    )
    parser.add_argument(
        "--spk_model",
        default="/opt/models/vosk-model-spk-0.4",
        help="Vosk speaker model path",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Ignore .npz cache and recompute embeddings",
    )
    parser.add_argument(
        "--out_json",
        default=None,
        help="Output json path (default: threshold_<split>.json)",
    )
    return parser.parse_args()


def build_speakers_struct(ds_dir: str, current_ds: str):
    # EXACT same variables/logic as notebook
    DS_DIR = ds_dir
    SPEAKERS_TXT = DS_DIR + "SPEAKERS.TXT"
    CHAPTERS_TXT = DS_DIR + "CHAPTERS.TXT"
    CURRENT_DS = current_ds

    speakers = {}

    # get the speakers that are in this dataset
    with open(SPEAKERS_TXT) as speakers_file:
        for speaker in speakers_file:
            if speaker[0] == ";":
                continue
            speaker_properties = speaker.split(" | ")
            speaker_ds = speaker_properties[2].split(" ")[0]
            if speaker_ds != CURRENT_DS:
                continue
            speaker_id = int(speaker_properties[0])
            speakers[speaker_id] = {"chapters": []}

    # add the chapters
    with open(CHAPTERS_TXT) as chapters:
        for chapter in chapters:
            if chapter[0] == ";":
                continue
            chapter_properties = chapter.split(" | ")
            speaker_id = int(chapter_properties[1])
            if speaker_id not in speakers:
                continue
            chapter_id = int(chapter_properties[0])
            speakers[int(speaker_id)]["chapters"] += [chapter_id]

    print(f"{CURRENT_DS} have {len(speakers)} speakers")

    speakers_df = [
        {"id": speaker_id, "chapters": chapters["chapters"]}
        for speaker_id, chapters in speakers.items()
    ]
    speakers_df = pd.DataFrame(speakers_df)
    return speakers, speakers_df, DS_DIR, CURRENT_DS


def ensure_dataset_from_gcs(ds_dir: str) -> str:
    """
    Ensure LibriSpeech metadata + dev-clean files are present locally.
    Your code expects ds_dir to contain SPEAKERS.TXT and CHAPTERS.TXT.

    This will download:
      gs://mlops-2026-dataset-bucket/dev-clean/LibriSpeech/
    into ds_dir.
    """
    ds_path = Path(ds_dir)
    ds_path.mkdir(parents=True, exist_ok=True)

    speakers_txt = ds_path / "SPEAKERS.TXT"
    chapters_txt = ds_path / "CHAPTERS.TXT"

    # Already present => do nothing
    if speakers_txt.exists() and chapters_txt.exists():
        return ds_dir

    # Download from GCS
    bucket = os.environ.get("GCS_BUCKET", "mlops-2026-dataset-bucket")
    prefix = os.environ.get("GCS_PREFIX", "dev-clean/LibriSpeech/")

    # Important: dest_dir is the folder that will contain SPEAKERS.TXT, CHAPTERS.TXT, and dev-clean/
    sync_gcs_prefix_to_dir(
        bucket_name=bucket,
        prefix=prefix,
        dest_dir=str(ds_path),
        skip_if_exists=True,
    )
    return ds_dir


def main():
    print("Running speaker embedding threshold computation...")
    args = parse_args()
    # If ds_dir does not contain LibriSpeech files, download them automatically from GCS
    args.ds_dir = ensure_dataset_from_gcs(args.ds_dir)
    if not args.ds_dir.endswith("/"):
        args.ds_dir += "/"

    # EXACT like notebook
    SetLogLevel(-1)

    model = Model(args.vosk_model)
    spk_model = SpkModel(args.spk_model)

    speakers, speakers_df, DS_DIR, CURRENT_DS = build_speakers_struct(
        args.ds_dir, args.current_ds
    )

    def samples_in_chapter(reader_id, chapter_id):
        chapter_dir = f"{DS_DIR}{CURRENT_DS}/{reader_id}/{chapter_id}"
        trans_filename = f"{reader_id}-{chapter_id}.trans.txt"
        trans_path = f"{chapter_dir}/{trans_filename}"
        with open(trans_path) as chapters_trans:
            for sample in chapters_trans:
                sample = sample.split(" ", 1)
                trans = sample[1].lower()
                sample_filename = sample[0]
                sample_path = f"{chapter_dir}/{sample_filename}.flac"
                yield (reader_id, chapter_id, sample_path, trans)

    def samples_of_speaker(reader_id):
        for chapter_id in speakers[reader_id]["chapters"]:
            yield from samples_in_chapter(reader_id, chapter_id)

    rec = KaldiRecognizer(model, 16000)
    rec.SetSpkModel(spk_model)

    def embed_speaker(filename):
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "quiet",
                "-i",
                filename,
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "s16le",
                "-",
            ],
            stdout=subprocess.PIPE,
        )

        embeddings = []
        while True:
            data = process.stdout.read(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                try:
                    embeddings += [result["spk"]]
                except KeyError:
                    pass

        result = json.loads(rec.Result())
        try:
            embeddings += [result["spk"]]
        except KeyError:
            pass

        return embeddings

    def get_speaker_embeddings(speaker_id):
        samples = list(samples_of_speaker(speaker_id))  # matérialise pour compter
        embeddings = []
        for _, _, sample_file, _ in tqdm(
            samples, desc=f"speaker {speaker_id}", unit="file", leave=False
        ):
            embeddings += embed_speaker(sample_file)
        return np.array(embeddings)

    # EXACT cache behavior as notebook (+ force flag)
    speakers_embeddings = []
    embedding_filename = f"speakers_embeddings_{CURRENT_DS}.npz"

    if not args.force_recompute:
        try:
            data = np.load(embedding_filename)
            speakers_embeddings = [data[f"arr_{i}"] for i in range(len(data.files))]
            print(f"Loaded {embedding_filename}")
        except FileNotFoundError:
            pass

    if len(speakers_embeddings) == 0:
        for _, speaker in tqdm(
            speakers_df.iterrows(),
            total=len(speakers_df),
            desc="embedding speakers",
            unit="speaker",
        ):
            speakers_embeddings += [get_speaker_embeddings(speaker["id"])]
        np.savez(embedding_filename, *speakers_embeddings)

    # IQR filtering EXACT like notebook
    nb_samples = np.array([emb.shape[0] for emb in speakers_embeddings])
    q1, q3 = np.percentile(nb_samples, [25, 75])
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    filtered_speakers_embeddings = [
        emb for emb in speakers_embeddings if lower <= emb.shape[0] <= upper
    ]

    print(
        f"removed {len(speakers_embeddings) - len(filtered_speakers_embeddings)} speakers"
    )

    def normalize(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    filtered_speakers_embeddings = [
        normalize(emb) for emb in filtered_speakers_embeddings
    ]

    # intra/inter EXACT like notebook
    intra_similarities = []
    inter_similarities = []

    for i, emb in enumerate(filtered_speakers_embeddings):
        sim_matrix = cosine_similarity(emb)

        # intra-similarities
        n = sim_matrix.shape[0]
        intra = sim_matrix[np.triu_indices(n, k=1)]
        intra_similarities.extend(intra)

        # inter-similarities
        for j in range(i + 1, len(filtered_speakers_embeddings)):
            emb_j = filtered_speakers_embeddings[j]
            cross_sim = cosine_similarity(emb, emb_j)
            inter_similarities.extend(cross_sim.ravel())

    intra_similarities = np.array(intra_similarities)
    inter_similarities = np.array(inter_similarities)

    # ROC + EER threshold EXACT like notebook (sans plots)
    scores = np.concatenate([intra_similarities, inter_similarities])
    labels = np.concatenate(
        [np.ones(len(intra_similarities)), np.zeros(len(inter_similarities))]
    )

    fpr, tpr, thresholds = roc_curve(
        labels, scores, pos_label=1, drop_intermediate=False
    )
    roc_auc = auc(fpr, tpr)

    eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
    eer = (fpr[eer_idx] + (1 - tpr[eer_idx])) / 2
    eer_threshold = thresholds[eer_idx]

    print("\n=== Threshold (EER) ===")
    print(f"AUC: {roc_auc:.4f}")
    print(f"EER: {eer:.4f}")
    print(f"Threshold: {eer_threshold:.4f}")

    # Save for training/inference
    out_json = args.out_json or f"threshold_{CURRENT_DS}.json"
    payload = {
        "split": CURRENT_DS,
        "threshold": float(eer_threshold),
        "eer": float(eer),
        "auc": float(roc_auc),
        "embedding_cache": embedding_filename,
        "n_speakers_total": int(len(speakers_embeddings)),
        "n_speakers_after_iqr": int(len(filtered_speakers_embeddings)),
        "n_intra_pairs": int(len(intra_similarities)),
        "n_inter_pairs": int(len(inter_similarities)),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()


"""    docker run --rm -it `
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/adc.json `
-e GOOGLE_CLOUD_PROJECT=info9023-project-hearedit `
-v "$env:APPDATA\\gcloud\\application_default_credentials.json:/secrets/adc.json:ro" `
ml-project uv run python speaker_embedding_threshold.py"""
