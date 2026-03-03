from __future__ import annotations

from pathlib import Path
from google.cloud import storage


def sync_gcs_prefix_to_dir(
    bucket_name: str,
    prefix: str,
    dest_dir: str,
    *,
    skip_if_exists: bool = True,
) -> None:
    """
    Download all objects under gs://bucket_name/prefix into dest_dir.

    - If skip_if_exists=True, files already present locally are not re-downloaded.
    - Keeps the folder structure relative to prefix.
    """
    client = storage.Client()
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    blobs = client.list_blobs(bucket_name, prefix=prefix)

    for blob in blobs:
        # GCS "folders" are just prefixes; sometimes a blob ends with "/" -> skip
        if blob.name.endswith("/"):
            continue

        rel_path = Path(blob.name).relative_to(prefix)
        out_path = dest / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if skip_if_exists and out_path.exists():
            continue

        blob.download_to_filename(str(out_path))