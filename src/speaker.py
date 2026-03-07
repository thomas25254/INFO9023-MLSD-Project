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

        self.similarity_policy = None
        if similarity_policy == "mean":
            self.similarity_policy = self.mean_embedding
        else:
            error_message = f"unknown similarity policy : {similarity_policy}"
            raise ValueError(error_message)

        self.history_policy = None
        if history_policy == "latest":
            self.history_policy  = self.latest_history
        else:
            raise ValueError(f"unknown history policy : {history_policy}")

        

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
        embeddings = [extract["speaker_embedding"] for extract in extracts]
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

        extract["speaker"] = self
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

        insert_at = 0
        for i, ext in enumerate(self.chronology):
            if extract["timestramp"] < ext["timestramp"]:
                insert_at = i
                break

        self.chronology.insert(insert_at, extract)
