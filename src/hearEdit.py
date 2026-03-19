from transcriber import Transcriber
from speaker import Speaker
from diarizer import Diarizer
from extractor import Extractor, Extract

import json


"""
TODO
----
- merge two extracts
- voir comment je peux utiliser a la fois result et final result
- correction de texte
- ~ apres correction de texte recomputing avec le texte connu
- ~ sliding windows pour le transcriber pour couper lorsque deux personnes parle sans laisser de blanc entre eux
- ~ essayer plusieurs policies pour la facon dont les prototypes sont faits
- ~ alternatives
- Je trouverai surement d'autres trucs en avancant

"""

class HearEdit:
    def __init__(self, threshold_path=None, model_path=None,
                 spk_model_path=None, audio_file=None):

        # the file to transcribe
        self.audio_file = audio_file

        # create the transcriber
        if model_path is not None and spk_model_path is not None and audio_file is not None:
            self.transcriber = Transcriber(model_path, spk_model_path, audio_file)
            self.transcription = self.transcriber.transcription()
            self.extractor = self.transcriber.extractor
        else:
            self.transcriber = None
            self.transcription = None
            self.extractor = None

        # create the diarizer
        if threshold_path is not None:
            self.diarizer = Diarizer(threshold_path=threshold_path)
        else:
            self.diarizer = None

        # extract is the dictionary linking the extract ids to the extracts
        self.extracts = {}
        # chronology is the chronological order of the extract ids
        self.chronology = []
        # timestamp is where the transcription stoped
        self.timestamp = 0.0


    def to_dict(self):
        return {"transcriber" : self.transcriber.to_dict(),
                "diarizer"    : self.diarizer.to_dict(),
                "extracts"    : {id_num : extract.to_dict() for id_num, extract
                                 in self.extracts.items()},
                "chronology"  : self.chronology,
                "timestamp"   : self.timestamp,
                }


    def to_json(self):
        return json.dumps(self.to_dict())


    @classmethod
    def from_dict(cls, data):
        hear_edit = cls()
        hear_edit.timestamp = data["timestamp"]

        # create the transcriber
        hear_edit.transcriber = Transcriber.from_dict(data["transcriber"])
        hear_edit.transcription = hear_edit.transcriber.transcription()
        hear_edit.extractor = hear_edit.transcriber.extractor

        # create the extracts, the speakers should be added after their
        # creation to each extract
        hear_edit.extracts = {int(extract_id) :
                              Extract.from_dict(extract, hear_edit.extractor)
                              for extract_id, extract in
                              data["extracts"].items()}
        hear_edit.chronology = data["chronology"]

        # create the diarizer, which creates the speakers
        hear_edit.diarizer = Diarizer.from_dict(data["diarizer"],
                                                hear_edit.extracts)
        # link the extracts to the speakers
        speakers = hear_edit.diarizer.speakers
        for extract_id, extract in data["extracts"].items():
            he_extract = hear_edit.extracts[int(extract_id)]
            he_extract.speaker = speakers[extract["speaker"]]

        return hear_edit


    @classmethod
    def from_json(cls, json_str):
        return cls.from_dict(json.loads(json_str))


    def set_audio_file(self, audio_file):
        self.audio_file = audio_file


    def open(self, file_path, at=0.0, to=0.0):
        self.transcriber.open(file_path, at, to)


    def play(self):
        # load where we were at
        if self.timestamp != self.transcriber.timestamp:
            self.transcriber.open(self.audio_file, self.timestamp)
        extract = next(self.transcription)
        self.timestamp = self.transcriber.timestamp
        self.diarizer.diarize(extract)
        self.extracts[extract.id] = extract
        self.chronology += [extract.id]
        return extract


    def split_extract(self, extract_id, at):
        # find extract
        extract = self.extracts[extract_id]
        # split it where it should
        new_extract = extract.split(at)

        # comptue embeddings of both individual extracts and assign it to
        # the correct speakers
        self.transcriber.open(self.audio_file, extract.start, extract.end)
        ext_emb = self.transcriber.transcribe()["spk"]
        extract.speaker_embedding = ext_emb

        self.transcriber.open(self.audio_file, new_extract.start, new_extract.end)
        ext_emb = self.transcriber.transcribe()["spk"]
        self.transcriber.timestamp = 0.0 # signify a reset when next play
        new_extract.speaker_embedding = ext_emb

        # self.diarizer.insert(new_extract, extract.speaker.name)
        self.diarizer.remove_extract_from_speaker(extract)
        self.diarizer.diarize(extract)
        self.diarizer.diarize(new_extract)

        # insert new one everywhere it should
        self.extracts[new_extract.id] = new_extract
        # find extract index in the chronology and add the new one next
        extract_at = 0
        for i, ext_id in enumerate(self.chronology):
            if ext_id == extract.id:
                extract_at = i
                break
        self.chronology.insert(extract_at + 1, new_extract.id)

        return new_extract


    def correct_speaker(self, extract_id, speaker_name):
        self.diarizer.correct_speaker(self.extracts[extract_id], speaker_name)


    def correct_text(self, extract_id, word_ranges, corrections):
        extract = self.extracts[extract_id]
        extract.correct_text(word_ranges, corrections)

    def rename_speaker(self, old_name, new_name):
        self.diarizer.rename_speaker(old_name, new_name)


    def print_chronology(self):
        for extract_id in self.chronology:
            print(self.extracts[extract_id])


    def print_speakers(self, print_extracts=False):
        for speaker_name, speaker in self.diarizer.speakers.items():
            print(speaker)



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
    """ outputs:
    (000000)[Copleston]{0.00-20.07} : if you offer steady already through
    experience a world that we've come to a knowledge of the existence that
    being and then one argues the essence and existence must be identical
    because if god's essence then god's existence were not identical then some
    sufficient reason for his existence would have to perform beyond god

    correct:
    It is only a posteriori through our experience of the world that we come to
    a knowledge of the existence of that being. And then one argues, the
    essence and existence must be identical. Because if God's essence and God's
    existence was not identical, then some sufficient reason for this existence
    would have to be found beyond God.
    """
    hear_edit.correct_text(extract.id, [[0,5], [6, 6] ], ["It is only a posteriori", "our"])


    # Then Russell
    extract = hear_edit.play()
    # But The transcriber took too much and some of Copelston response got in
    new_extract = hear_edit.split_extract(extract.id, 37)
    # rename the first
    hear_edit.rename_speaker(extract.speaker.name, "Russell")

    # Copleston
    extract = hear_edit.play()


    # Russell
    extract = hear_edit.play()

    # save to json and then reload the HE instance
    he_str = hear_edit.to_json()
    hear_edit = HearEdit.from_json(he_str)
    he2_str = hear_edit.to_json()
    print("jsoned hear_edit and reloaded it, both json string are the same:",
          he_str == he2_str)

    hear_edit.set_audio_file(test_file)

    # Russell
    extract = hear_edit.play()
    hear_edit.correct_speaker(extract.id, "Russell")

    # End of the extract. Should send a StopIteration error
    try:
        extract = hear_edit.play()
    except StopIteration:
        print("end of the transcription")

    print()
    hear_edit.print_chronology()

    print()
    hear_edit.print_speakers()
