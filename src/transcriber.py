import json
import os
import subprocess
import tempfile

from vosk import KaldiRecognizer, Model, SetLogLevel
from train_speaker_pair_nn import load_finetuned_encoder_only, extract_embedding_from_audio

from extractor import Extractor

FFMPEG_PATH = "C:\\Users\\alexa\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe"
class Transcriber:
    def __init__(self, model_path, spk_model_path, file_path=None):
        SetLogLevel(-1)
        self.model_path = model_path
        self.spk_model_path = spk_model_path  # plus vraiment utilisé
        self.sb_encoder = load_finetuned_encoder_only(spk_model_path)
        self.model = Model(model_path)
        self.timestamp = 0.0
        self.rec = None
        self.extractor = Extractor()
        self.wav = None
        self.file_path = None
        self.start_at = 0.0

        if file_path is not None:
            self.open(file_path)

    def to_dict(self):
        return {
            "model_path": self.model_path,
            "spk_model_path": self.spk_model_path,
            "extractor": self.extractor.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        transcriber = cls(data["model_path"], data["spk_model_path"])
        transcriber.extractor = Extractor.from_dict(data["extractor"])
        transcriber.timestamp = 0.0
        return transcriber
    
    def open(self, file_path, at=0.0, to=0.0):
        self.file_path = file_path
        self.rec = KaldiRecognizer(self.model, 16000)
        self.rec.SetWords(True)

        process_str = [
            FFMPEG_PATH,
            "-loglevel",
            "quiet",
            "-i",
            file_path,
        ]

        if at != 0.0:
            process_str += ["-ss", str(at)]
        if to != 0.0:
            process_str += ["-to", str(to)]

        process_str += [
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "s16le",
            "-",
        ]

        self.wav = subprocess.Popen(process_str, stdout=subprocess.PIPE)
        self.start_at = at

    def compute_segment_embedding(self, seg_start, seg_end):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        process_str = [
            FFMPEG_PATH,
            "-loglevel", "quiet",
            "-y",
            "-i", self.file_path,
            "-ss", str(seg_start),
            "-to", str(seg_end),
            "-ar", "16000",
            "-ac", "1",
            tmp_path,
        ]
        subprocess.run(process_str, check=True)

        emb = extract_embedding_from_audio(self.sb_encoder, tmp_path)
        os.remove(tmp_path)

        return emb.tolist()

    def add_speaker_embedding(self, result):
        if "result" in result and len(result["result"]) > 0:
            seg_start = result["result"][0]["start"] + self.start_at
            seg_end = result["result"][-1]["end"] + self.start_at
            result["spk"] = self.compute_segment_embedding(seg_start, seg_end)
            self.timestamp = seg_end
        return result

    def transcribe(self):
        while True:
            data = self.wav.stdout.read(4000)
            if len(data) == 0:
                break

            if self.rec.AcceptWaveform(data):
                result = json.loads(self.rec.Result())
                return self.add_speaker_embedding(result)

        if self.wav.poll() is None:
            self.wav.terminate()
            self.wav.wait()

        result = json.loads(self.rec.FinalResult())
        return self.add_speaker_embedding(result)

    def transcription(self):
        while True:
            data = self.wav.stdout.read(4000)
            if len(data) == 0:
                break

            if self.rec.AcceptWaveform(data):
                result = json.loads(self.rec.Result())
                result = self.add_speaker_embedding(result)

                if "result" in result and len(result["result"]) > 0:
                    extract = self.extractor.new_extract(result, self.start_at)
                    self.timestamp = extract.end
                    yield extract

        if self.wav.poll() is None:
            self.wav.terminate()
            self.wav.wait()

        result = json.loads(self.rec.FinalResult())
        result = self.add_speaker_embedding(result)

        if "result" in result and len(result["result"]) > 0:
            extract = self.extractor.new_extract(result, self.start_at)
            self.timestamp = extract.end
            yield extract


if __name__ == "__main__":
    model_path = "C:\\Users\\alexa\\OneDrive - Universite de Liege\\University\\2025-2026\\Q2\\MLSD\\Project\\vosk-model-en-us-0.22"
    test_file = "C:\\Users\\alexa\\OneDrive - Universite de Liege\\University\\2025-2026\\Q2\\MLSD\\Project\\debate_extract.wav"
    spk_model_path = "C:\\Users\\alexa\\OneDrive - Universite de Liege\\University\\2025-2026\\Q2\\MLSD\\Project\\artifacts\\ecapa_finetuned_speakerid_hidden512.pt"
    FFMPEG_PATH = "C:\\Users\\alexa\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe"
    transcriber = Transcriber(model_path, file_path=test_file, spk_model_path=spk_model_path)

    for trans in transcriber.transcription():
        emb = trans.speaker_embeddings
        text = trans.text()
        print(f"embedding_dim={len(emb)} | first5={emb[:5]} | text={text}")