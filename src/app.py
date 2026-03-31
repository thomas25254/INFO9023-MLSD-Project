import os
import sys
import tempfile

# src/ is already the working directory when launched from there,
# but make it explicit so imports work regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, render_template, request

from hearEdit import HearEdit

app = Flask(__name__)


def _short(path):
    """Return Windows short (8.3) path to avoid accent issues with Vosk."""
    if os.name == "nt":
        import ctypes

        buf = ctypes.create_unicode_buffer(500)
        ctypes.windll.kernel32.GetShortPathNameW(path, buf, 500)
        return buf.value or path
    return path


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.dirname(os.path.abspath(__file__))

# GCS config — set these env vars in Cloud Run to enable automatic model download
GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
GCS_MODELS_PREFIX = os.environ.get("GCS_MODELS_PREFIX", "models/vosk-model-en-us-0.22")
GCS_ARTIFACT_PREFIX = os.environ.get(
    "GCS_ARTIFACT_PREFIX", "artifacts/ecapa_finetuned_speakerid_hidden512.pt"
)

# Local paths where models are expected (or will be downloaded to)
_MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(_ROOT, "models"))
_ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", os.path.join(_ROOT, "artifacts"))

THRESHOLD_PATH = _short(
    os.environ.get(
        "THRESHOLD_PATH",
        os.path.join(_SRC, "threshold_dev-clean.json"),
    )
)
MODEL_PATH = _short(
    os.environ.get(
        "MODEL_PATH",
        os.path.join(_MODELS_DIR, "vosk-model-en-us-0.22"),
    )
)
SPK_MODEL_PATH = _short(
    os.environ.get(
        "SPK_MODEL_PATH",
        os.path.join(_ARTIFACTS_DIR, "ecapa_finetuned_speakerid_hidden512.pt"),
    )
)


def _download_models_from_gcs():
    """Download model files from GCS if not already present locally."""
    if not GCS_BUCKET:
        return
    from gcs_download import sync_gcs_prefix_to_dir

    if not os.path.isdir(MODEL_PATH):
        print(f"Downloading Vosk model from gs://{GCS_BUCKET}/{GCS_MODELS_PREFIX} ...")
        sync_gcs_prefix_to_dir(GCS_BUCKET, GCS_MODELS_PREFIX, _MODELS_DIR)
        print("Vosk model ready.")

    if not os.path.isfile(SPK_MODEL_PATH):
        artifact_dir = os.path.dirname(SPK_MODEL_PATH)
        artifact_prefix = GCS_ARTIFACT_PREFIX
        print(f"Downloading speaker model from gs://{GCS_BUCKET}/{artifact_prefix} ...")
        sync_gcs_prefix_to_dir(GCS_BUCKET, artifact_prefix, artifact_dir)
        print("Speaker model ready.")


_download_models_from_gcs()

_past_transcriptions = []


def transcribe_audio(audio_path):
    """Run HearEdit on audio_path, return list of segments with speaker info."""
    hear_edit = HearEdit(THRESHOLD_PATH, MODEL_PATH, SPK_MODEL_PATH, audio_path)
    hear_edit.extractor.set_timestamp_format(True)

    # Read remaining segments until end
    try:
        while True:
            hear_edit.play()
    except StopIteration:
        pass

    # Collect all segments from chronology
    segments = [
        {
            "start": hear_edit.extracts[extract_id].start,
            "end": hear_edit.extracts[extract_id].end,
            "speaker": hear_edit.extracts[extract_id].speaker.name
            if hear_edit.extracts[extract_id].speaker
            else "unknown",
            "text": hear_edit.extracts[extract_id].text(),
        }
        for extract_id in hear_edit.chronology
    ]

    return segments


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify(
            {
                "error": "No audio file provided. Send a multipart request with field 'audio'."
            }
        ), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    suffix = os.path.splitext(audio_file.filename)[1] or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = _short(tmp.name)

        segments = transcribe_audio(tmp_path)
        full_text = " ".join(s["text"] for s in segments)
        result = {
            "filename": audio_file.filename,
            "segments": segments,
            "full_text": full_text,
        }
        _past_transcriptions.append(result)
        return jsonify(result)

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/past_transcriptions", methods=["GET"])
def past_transcriptions():
    return jsonify(
        {
            "count": len(_past_transcriptions),
            "transcriptions": [
                {
                    "id": i,
                    "filename": t["filename"],
                    "full_text": t["full_text"],
                }
                for i, t in enumerate(_past_transcriptions)
            ],
        }
    )


@app.route("/past_transcriptions/<int:transcription_id>", methods=["GET"])
def get_transcription(transcription_id):
    if transcription_id >= len(_past_transcriptions):
        return jsonify({"error": "Not found"}), 404
    return jsonify(_past_transcriptions[transcription_id])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)
