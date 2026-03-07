import json
import subprocess
from vosk import KaldiRecognizer, Model, SpkModel, SetLogLevel
from extractor import Extractor



class Transcriber:
    def __init__(self, model_path, spk_model_path, file_path):
        SetLogLevel(-1)
        self.model = Model(model_path)
        self.spk_model = SpkModel(spk_model_path)
        self.rec = KaldiRecognizer(self.model, 16000)
        self.rec.SetSpkModel(self.spk_model)
        self.extractor = Extractor()
        # self.rec.SetWords(True)
        self.wav = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "quiet",
                "-i",
                file_path,
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


    def transcription(self):
        while True:
            data = self.wav.stdout.read(4000)
            if len(data) == 0:
                break
            if self.rec.AcceptWaveform(data):
                result = json.loads(self.rec.Result())
                yield self.extractor.new_extract(result)

        if self.wav.poll() == None:
            self.wav.terminate()
            self.wav.wait()
        result = json.loads(self.rec.Result())
        yield self.extractor.new_extract(result)

if __name__ == "__main__":
    model_path = "../../../vosk-model-en-us-0.22"
    spk_model_path = "../../../vosk-model-spk-0.4"
    test_file = "../../../data/debate_extract.wav"
    transcriber = Transcriber(model_path, spk_model_path, test_file)
    for trans in transcriber.transcription():
        emb = trans['speaker_embedding']
        text = trans['text']
        print(f"[{emb[0]:2f}, ..., {emb[0]:2f}] : {text}")
