from transcriber import Transcriber
from speaker import Speaker
from diarizer import Diarizer
from extractor import Extractor


"""
TODO
----

- recomputing lors du split
- corrections de texte et recomputing
- sliding windows pour le transcriber pour couper lorsque deux personnes parle sans laisser de blanc entre eux
-  ~ essayer plusieurs policies pour la facon dont les prototypes sont faits
- faire un fichier de sauvegarde
- voir comment je peux utiliser a la fois result et final result
- ~ alternatives
- Je trouverai surement d'autres trucs en avancant

"""

class HearEdit:
    def __init__(self, threshold_path, model_path, spk_model_path, audio_file):
        self.transcriber = Transcriber(model_path, spk_model_path, audio_file)
        self.transcription = self.transcriber.transcription()
        self.extractor = self.transcriber.extractor
        self.diarizer = Diarizer(threshold_path)
        self.extracts = {}
        self.chronology = []


    def play(self):
        extract = next(self.transcription)
        self.diarizer.diarize(extract)
        self.extracts[extract.id] = extract
        self.chronology += [extract]
        return extract


    def split_extract(self, extract_id, at):
        # find extract
        extract = self.extracts[extract_id]
        # split it where it should
        new_extract = extract.split(at)

        # TODO comptue embeddings of both individual extracts and assign it to
        # the correct speakers

        # insert it everywhere it should
        self.diarizer.insert(new_extract, extract.speaker.name)
        self.extracts[new_extract.id] = new_extract
        # find extract index in the chronology and add the new one next
        extract_at = 0
        for i, ext in enumerate(self.chronology):
            if ext == extract:
                extract_at = i
                break
        self.chronology.insert(extract_at + 1, new_extract)

        return new_extract


    def correct_speaker(self, extract_id, speaker_name):
        self.diarizer.correct_speaker(self.extracts[extract_id], speaker_name)


    def rename_speaker(self, old_name, new_name):
        self.diarizer.rename_speaker(old_name, new_name)


    def print_chronology(self):
        for extract in self.chronology:
            print(extract)


    def print_speakers(self, print_extracts=False):
        for speaker_name, speaker in self.diarizer.speakers.items():
            print(speaker)
            if print_extracts:
                for extract in speaker.chronology:
                    print(f"\t{extract}")
                print()



if __name__ == "__main__":
    # hear_edit
    threshold_path = "threshold_dev-clean.json"
    model_path = "../../../vosk-model-en-us-0.22"
    spk_model_path = "../../../vosk-model-spk-0.4"
    test_file = "../../../data/debate_extract.wav"
    hear_edit = HearEdit(threshold_path, model_path, spk_model_path, test_file)


    hear_edit.extractor.set_timestamp_format(True)
    hear_edit.extractor.set_id_format(True)


    # First Copleston speaks
    extract = hear_edit.play()

    hear_edit.rename_speaker(extract.speaker.name, "Copleston")

    # Then Russell
    extract = hear_edit.play()
    hear_edit.rename_speaker(extract.speaker.name, "Russell")

    # But The transcriber took too much and some of Copelston response got in
    new_extract = hear_edit.split_extract(extract.id, 37)
    hear_edit.correct_speaker(new_extract.id, "Copleston")

    # Copleston
    extract = hear_edit.play()

    # Russell
    extract = hear_edit.play()
    # Russel is not detected because he has no embedding yet. TODO get embedding when splitting
    hear_edit.correct_speaker(extract.id, "Russell")

    # Russell
    extract = hear_edit.play()
    hear_edit.correct_speaker(extract.id, "Russell")

    # End of the extract. Should send a StopIteration error
    try:
        extract = hear_edit.play()
    except StopIteration:
        pass

    print()
    hear_edit.print_chronology()

    print()
    hear_edit.print_speakers()
