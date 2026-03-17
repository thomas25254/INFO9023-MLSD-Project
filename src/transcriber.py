import json
import subprocess
from vosk import KaldiRecognizer, Model, SpkModel, SetLogLevel
from extractor import Extractor



class Transcriber:
    def __init__(self, model_path, spk_model_path, file_path=None):
        SetLogLevel(-1)
        self.model_path = model_path
        self.spk_model_path = spk_model_path
        self.model = Model(model_path)
        self.spk_model = SpkModel(spk_model_path)
        self.timestamp = 0.0
        self.rec = None
        self.extractor = Extractor()
        self.wav = None
        if file_path is not None:
            self.open(file_path)


    def to_dict(self):
        return {"model_path"     : self.model_path,
                "spk_model_path" : self.spk_model_path,
                "extractor"      : self.extractor.to_dict(),
                }

    @classmethod
    def from_dict(cls, data):
        transcriber = cls(data["model_path"], data["spk_model_path"])
        transcriber.extractor = Extractor.from_dict(data["extractor"])
        transcriber.timestamp = 0.0
        return transcriber


    def open(self, file_path, at=0.0, to=0.0):
        
        # recreate the model
        self.rec = KaldiRecognizer(self.model, 16000)
        self.rec.SetSpkModel(self.spk_model)
        self.rec.SetWords(True)
        # self.rec.SetMaxAlternatives(3)

        # create the process to turn the file to raw wav form
        process_str = ["ffmpeg",
                       "-loglevel", "quiet",
                       "-i", file_path,
                       ]
        if at != 0.0:
            process_str += ["-ss", str(at)]

        if to != 0.0:
            process_str += ["-to", str(to)]

        process_str += ["-ar", "16000",
                        "-ac", "1",
                        "-f", "s16le",
                        "-",
                        ]

        self.wav = subprocess.Popen(process_str, stdout=subprocess.PIPE)
        self.start_at = at


    def transcribe(self):
        while True:
            data = self.wav.stdout.read(4000)
            if len(data) == 0:
                break
            if self.rec.AcceptWaveform(data):
                result = json.loads(self.rec.Result())
                self.timestamp = result["result"][-1]["end"] + self.start_at
                return result

        if self.wav.poll() is None:
            self.wav.terminate()
            self.wav.wait()
        result = json.loads(self.rec.Result())
        self.timestamp = result["result"][-1]["end"] + self.start_at
        return result


    def transcription(self):
        while True:
            data = self.wav.stdout.read(4000)
            if len(data) == 0:
                break
            if self.rec.AcceptWaveform(data):
                result = json.loads(self.rec.Result())
                extract = self.extractor.new_extract(result, self.start_at)
                self.timestamp = extract.end
                yield extract

        if self.wav.poll() is None:
            self.wav.terminate()
            self.wav.wait()
        result = json.loads(self.rec.Result())
        extract =  self.extractor.new_extract(result, self.start_at)
        self.timestamp = extract.end
        yield extract


if __name__ == "__main__":
    model_path = "../../../vosk-model-en-us-0.22"
    spk_model_path = "../../../vosk-model-spk-0.4"
    test_file = "../../../data/debate_extract.wav"
    transcriber = Transcriber(model_path, spk_model_path, test_file)
    for trans in transcriber.transcription():
        emb = trans.speaker_embedding
        text = trans.text()
        print(f"[{emb[0]:2f}, ..., {emb[0]:2f}] : {text}")
