# Vosk Speaker Recognition Data Exploration


## What is Vosk?

_Vosk_ is an offline speech to text library that can work in python. It is composed of an accoustic neural model followed by a language model. It also provides a voice feature extractor.

## Purpose of this notebook

This notebook aims at finding a good way to discriminate speakers based on their voices. The algorithm should be able to determine when a new, never seen speaker comes into play and distinguish the different speaker.

## Method used

Multiple solutions were considered like a one-shot neural network but the simplest and most reliable solution seems to be the cosine similarity method.

From the extracted voice features vector a cosine similarity value is computed from every known existing speakers' prototype voice feature vector. If one value is smaller than a certain threshold it is considered as spoken by this speaker. If multiple values are below this threshold it is assigned to the speaker with the smallest difference. If no values are below this threshold then a new speaker is created with the prototype being this vector. Finally the speaker prototype vector is updated with this new vector and is the mean of all its known vectors (TODO maybe not all the vectors, to determine later, not during the "training").

The last thing needed is the value of that threshold. It should be determined empirically, learned from some training data. This notebook will analyse the intra-speaker and extra-speaker cosine similarity value from a small dataset and test multiple threshold extracted from a statistical analysis of those values to determine the best disciminative threshold.
## Dataset
The dataset used here is the [LibriSpeech dev-clean](https://openslr.org/12/) one. It is the smallest one used especially for tests and data exploration.
# Result
The chosen threshold is the EER (Equal Error Rate) to have an equal rate of false acceptance and false rejection.

The intra and inter speaker voice feature cosine similarity distribution overlaps quite a bit. The selection of EER gives a 9.5% error on FAR and FRR.
![enter image description here](https://github.com/thomas25254/INFO9023-MLSD-Project/blob/dev/model_training_cleaning/notebooks/model_training/vosk_speaker_recognition/app/speaker_similarity_distributions.png)

## How to Run this Notebook

build the container:\
`docker compose build`


start the container:\
`docker compose up`
