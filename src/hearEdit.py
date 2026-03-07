from transcriber import Transcriber
from speaker import Speaker
from diarizer import Diarizer



class HearEdit(Diarizer):
    def __init__(self, threshold_path, model_path, spk_model_path, audio_file):
        super(HearEdit, self).__init__(threshold_path)
        self.transcriber = Transcriber(model_path, spk_model_path, audio_file)
        self.transcription = self.transcriber.transcription()


    def play(self):
        extract = next(self.transcription)
        self.diarize(extract)
        return extract




if __name__ == "__main__":
    # hear_edit
    threshold_path = "threshold_dev-clean.json"
    model_path = "../../../vosk-model-en-us-0.22"
    spk_model_path = "../../../vosk-model-spk-0.4"
    test_file = "../../../data/debate_extract.wav"
    hear_edit = HearEdit(threshold_path, model_path, spk_model_path, test_file)


    # First Copleston speaks
    extract = hear_edit.play()
    hear_edit.rename_speaker(extract["speaker"].name, "Copleston")
    print(f"[{extract["speaker"].name}] : {extract["text"]}")

    # Then Russell
    extract = hear_edit.play()
    hear_edit.rename_speaker(extract["speaker"].name, "Russell")
    # TODO But The transcriber took too much and some of Copelston response got in
    print(f"[{extract["speaker"].name}] : {extract["text"]}")

    # Copleston
    extract = hear_edit.play()
    print(f"[{extract["speaker"].name}] : {extract["text"]}")

    # Russel
    extract = hear_edit.play()
    print(f"[{extract["speaker"].name}] : {extract["text"]}")

    # Russel
    extract = hear_edit.play()
    print(f"[{extract["speaker"].name}] : {extract["text"]}")

    # End of the extract. Should send a StopIteration error
    extract = hear_edit.play()
    print(f"[{extract["speaker"].name}] : {extract["text"]}")
