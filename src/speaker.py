from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



class Speaker:
    def __init__(self, name, similarity_policy="mean", max_hist=10,
                 history_policy="latest"):
        """Creates a new speaker

        Parameters
        ----------
        name : string
            The name of the speaker
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
        Speaker
            The created speaker
        """

        self.name = name
        # self.embeddings = [np.array(embedding)]
        self.chronology = []
        self.max_hist = max_hist

        self.similarity_policy_str = similarity_policy
        self.similarity_policy = None
        if similarity_policy == "mean":
            self.similarity_policy = self.mean_embedding
        else:
            error_message = f"unknown similarity policy : {similarity_policy}"
            raise ValueError(error_message)

        self.history_policy_str = history_policy
        self.history_policy = None
        if history_policy == "latest":
            self.history_policy  = self.latest_history
        else:
            raise ValueError(f"unknown history policy : {history_policy}")


    def to_dict(self):
        """gives the speaker state in a dict

        Returns
        -------
        dict
            the states of the speaker
        """

        return {"name"              : self.name,
                "chronology"        : [extract.id for extract in
                                       self.chronology],
                "max_hist"          : self.max_hist,
                "similarity_policy" : self.similarity_policy_str,
                "history_policy"    : self.history_policy_str,
                }

    @classmethod
    def from_dict(cls, data, extracts):
        """Creates a new speaker from a dict of its states

        Parameters
        ----------
        data : dict
            states of the speaker
        extracts : dict
            a dict linking exdtracts ids to extracts

        Returns
        -------
        speaker : Speaker
            a speaker with the states in data
        """

        speaker = cls(data["name"], data["similarity_policy"],
                      data["max_hist"], data["history_policy"])
        speaker.chronology = [extracts[ext_id] for ext_id in data["chronology"]]
        return speaker

    def similarity(self, embedding):
        """Computes the similarity according to the similarity policy"

        Parameters
        ----------
        embedding : list or numpy array
            The embedding to compare to

        Returns
        -------
        similarity : 2D numpy array
            The similarity value according to the similarity policy
        """

        return self.similarity_policy(embedding)


    def mean_embedding(self, embedding):
        """Computes the similarity based on the mean of the history of known
        embeddings

        Parameters
        ----------
        embedding : list or numpy array
            The embedding to compare to

        Returns
        -------
        similarity : 2D numpy array
            The similarity value based on the mean of the history of known
            embeddings
        """

        # take the embeddings according to the history policy
        extracts = self.history_policy()
        embeddings = [emb for extract in extracts if extract.speaker_embeddings
                      is not None for emb in extract.speaker_embeddings]
        if len(embeddings) == 0:
            raise ValueError("no previous embedding")
        # do the mean and compute similarity
        mean_embedding =  np.array(embeddings).mean(axis=0)
        similarity = cosine_similarity(mean_embedding.reshape(1, -1),
                                       np.array(embedding).reshape(1, -1))
        return similarity


    def update(self, extract):
        """Updates the speaker prototype based on the history policy

        Parameters
        ----------
        extract : dict
            The extract to update the prototype with
        """

        extract.speaker = self
        self.chronology += [extract]
        return

    def latest_history(self):
        """Updates the speaker prototype by keeping only the latest embeddings
        """

        return self.chronology[-self.max_hist:]


    def insert(self, extract):
        """Insert the extract at the right time in the chronology

        Parameters
        ----------
        extract : dict
            The extract to update the prototype with
        """

        # if none goes after this one then insert it last
        insert_at = len(self.chronology)
        for i, ext in enumerate(self.chronology):
            # insert it where the nearest but superior in time is
            if extract.start <= ext.start:
                insert_at = i + 1
                break

        self.chronology.insert(insert_at, extract)
        extract.speaker = self


    def __str__(self):
        embeddings = [emb for extract in self.chronology if extract.speaker_embeddings
                      is not None for emb in extract.speaker_embeddings]
        return f"{self.name} : {len(embeddings)} embeddings\n\textracts : {[extract.id for extract in self.chronology]}"
