import json
import numpy as np
from transcriber import Transcriber
from speaker import Speaker



class Diarizer:
    # def __init__(self, threshold_path, threshold=True):
    def __init__(self, threshold=None, threshold_path=None,
                 similarity_policy="mean", max_hist=10,
                 history_policy="latest"):
        """Creates a new Diarizer

        Parameters
        ----------
        threshold_path : string
            a path to the json fiole containing the threshold value
        TODO threshold : boolean
            Either to use a threshold or a NN model
        similarity_policy : string, "mean" TODO other like KNN
            policy to compute the similarity
                - "mean" compare the new embedding to the mean of the known
                  ones
                - TODO ""KNN" compare to every new embedding and returns the
                  one with the most similarity
        max_hist : int
            The maximum number of embeddings to have in history
        history_policy : string, "latest" TODO "furthest"
            policy to keep which embedding
            "latest" keeps only the latest embeddings
            TODO "furthest" keep the embeddings the furthest apart

        Returns
        -------
        Diarizer
            The created diarizer
        """

        self.similarity_policy = similarity_policy
        self.max_hist = max_hist
        self.history_policy = history_policy

        self.speakers = {}

        self.threshold_path = threshold_path
        if threshold is not None:
            self.threshold = threshold
        elif threshold_path is not None:
            with open(threshold_path) as threshold_file:
                data = json.load(threshold_file)
                self.threshold = data['threshold']
        else:
            raise ValueError("Either give a threshold or a threshold_path")


    def to_dict(self):
        return {"similarity_policy" : self.similarity_policy,
                "history_policy"    : self.history_policy,
                "max_hist"          : self.max_hist,
                "threshold"         : self.threshold,
                "threshold_path"    : self.threshold_path,
                "speakers"          : [speaker.to_dict() for _, speaker in
                                       self.speakers.items()],
               }

    @classmethod
    def from_dict(cls, data, extracts):
        """Creates a Diarizer with data states

        Parameters
        ----------
        data : dict
            states for the diarizer
        extracts : dict
            a dict linking exdtracts ids to extracts

        Returns
        -------
        diarizer : Diarizer
            a diarizer with data states
        """

        diarizer = cls(data["threshold"], data["threshold_path"],
                       data["similarity_policy"], data["max_hist"],
                       data["history_policy"])
        diarizer.speakers = {speaker["name"] : Speaker.from_dict(speaker,
                                                                 extracts) for
                             speaker in data["speakers"]}
        return diarizer

    def diarize(self, extract):
        """Find who is the speaker or if it is a new one

        Parameters
        ----------
        extract : dict
            The extract to diarize

        Returns
        -------
        speaker : Speaker
            The corresponding speaker or the new one
        """

        # get the similarities between this embedding and the speakers prototypes
        embedding = extract.speaker_embeddings[0]
        similarities = {}
        for _, speaker in self.speakers.items():
            try:
                similarities[speaker] = speaker.similarity(embedding)
            except ValueError:
                similarities[speaker] = np.array([[0.0]])

        # find the most similar
        most_similar_speaker = (None, 0)
        for similarity in similarities.items():
            if most_similar_speaker[1] < similarity[1][0,0]:
                most_similar_speaker = (similarity[0], similarity[1][0,0])

        # if there are no speakers or if the one with biggest similarity is
        # smaller than the threshold create a new one
        if most_similar_speaker[0] is None or (most_similar_speaker[1] <
                                               self.threshold):

            # create a new speaker name that do not already exists
            new_speaker_nb = len(self.speakers) + 1
            new_speaker_name = f"speaker {new_speaker_nb}"
            while new_speaker_name in self.speakers:
                new_speaker_nb += 1
                new_speaker_name = f"speaker {new_speaker_nb}"

            # create the new speaker, update it and return it
            speaker = Speaker(new_speaker_name, self.similarity_policy,
                              self.max_hist, self.history_policy)
            self.speakers[new_speaker_name] = speaker
            speaker.update(extract)
            return speaker

        else:
            most_similar_speaker[0].update(extract)
            return most_similar_speaker[0]


    def rename_speaker(self, old_name, new_name):
        """Rename a speaker from an old to a new name

        Parameters
        ----------
        old_name : str
            The old name of the speaker
        new_name : str
            The new name of the speaker
        """

        self.speakers[old_name].name = new_name
        self.speakers[new_name] = self.speakers[old_name]
        del self.speakers[old_name]


    def remove_extract_from_speaker(self, extract):
        # remove from the old speaker and remove the speaker if it is empty
        old_speaker = extract.speaker
        old_speaker_name = old_speaker.name
        old_speaker.chronology.remove(extract)
        if len(old_speaker.chronology) == 0:
            del self.speakers[old_speaker_name]


    def correct_speaker(self, extract, speaker_name):
        """Correct the speaker of an extract

        Parameters
        ----------
        extract : Extract
            The extract to correct
        speaker_name : str
            The name of the correct speaker
        """

        # remove from the old one
        self.remove_extract_from_speaker(extract)
        # add it to the new one
        self.insert(extract, speaker_name)

        return

    def insert(self, extract, speaker_name):
        """Inserts the extract in this speaker

        Parameters
        ----------
        extract : Extract
            The extract to correct
        speaker_name : str
            The name of the speaker to insert the extract to
        """

        # add it to the new speaker and create the speaker if it doesn't exist
        speaker = None
        try:
            speaker = self.speakers[speaker_name]
        except KeyError:
            speaker = Speaker(new_speaker_name, self.similarity_policy,
                              self.max_hist, self.history_policy)
            self.speakers[speaker_name] = speaker
        speaker.insert(extract)



if __name__ == "__main__":
    # diarizer
    threshold_path = "threshold_dev-clean.json"
    diarizer = Diarizer(threshold_path=threshold_path)

    # transcriber
    model_path = "../../../vosk-model-en-us-0.22"
    spk_model_path = "../../../vosk-model-spk-0.4"
    test_file = "../../../data/debate_extract.wav"
    transcriber = Transcriber(model_path, spk_model_path, test_file)

    # transcribe and diarize speaker step by step
    for extract in transcriber.transcription():
        speaker = diarizer.diarize(extract)
        print(f"[{extract.speaker.name}] : {extract.text()}")
