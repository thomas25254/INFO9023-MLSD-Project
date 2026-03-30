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

THRESHOLD_PATH = _short(
    os.environ.get(
        "THRESHOLD_PATH",
        os.path.join(_SRC, "threshold_dev-clean.json"),
    )
)
MODEL_PATH = _short(
    os.environ.get(
        "MODEL_PATH",
        os.path.join(_ROOT, "models", "vosk-model-en-us-0.22"),
    )
)
SPK_MODEL_PATH = _short(
    os.environ.get(
        "SPK_MODEL_PATH",
        os.path.join(_ROOT, "artifacts", "ecapa_finetuned_speakerid_hidden512.pt"),
    )
)

_past_transcriptions = []


def transcribe_audio(audio_path):
    """Run HearEdit on audio_path, return list of segments with speaker info."""
    hear_edit = HearEdit(THRESHOLD_PATH, MODEL_PATH, SPK_MODEL_PATH, audio_path)
    hear_edit.extractor.set_timestamp_format(True)

    # To remove --------------
    hear_edit.extractor.set_id_format(True)

    # First Copleston speaks
    extract = hear_edit.play()
    hear_edit.rename_speaker(extract.speaker.name, "Copleston")
    hear_edit.correct_text(
        extract.id, [([0, 5], "It is only a posteriori"), ([6, 6], "our")]
    )

    # Then Russell
    extract = hear_edit.play()
    # But The transcriber took too much and some of Copelston response got in
    hear_edit.split_extract(extract.id, 37)
    # rename the first
    hear_edit.rename_speaker(extract.speaker.name, "Russell")

    # Copleston
    extract = hear_edit.play()
    hear_edit.merge_extract_with_preceding(extract.id)

    # Russell
    extract = hear_edit.play()

    # save to json and then reload the HE instance
    he_str = hear_edit.to_json()
    hear_edit = HearEdit.from_json(he_str)

    hear_edit.set_audio_file(audio_path)

    # Russell
    extract = hear_edit.play()
    hear_edit.correct_speaker(extract.id, "Russell")
    hear_edit.merge_extract_with_preceding(extract.id)

    # ------------------

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
    app.run(debug=True, host="0.0.0.0", port=5000)
