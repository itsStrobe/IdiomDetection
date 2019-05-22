"""
    File:   TrainWordEmbeddings
    Author: Jose Juan Zavala Iglesias
    Date:   21/02/2019

    Train Word Embedding Models with BNC XML Corpora and Store them for later use.
"""
import numpy as np
from Util import CorpusExtraction
from WordEmbeddings import Word2Vec_Embeddings as W2V

CORPORA_PRE  = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K"]
MODEL_DIR    = "./models/"
MODEL_SUFFIX = "_ver1.model"

# Following values by King and Cook (2018)
VEC_SIZE = 300
WND_SIZE = 8
EPOCHS   = 5

initialize_models = True

model_W2V = None
model_SCB = None
model_SkT = None
model_ELM = None

for corpora_name in CORPORA_PRE:
    print("Loading Corpora:", corpora_name)
    corpora = CorpusExtraction.LoadCorpora(corpora_name)

    for corpus in corpora:
        print("Training Models on Corpus:", corpus)
        sentences = corpora[corpus].tolist()

        if (initialize_models):

            # Initializing Word2Vec
            model_W2V = W2V(sentences, size=VEC_SIZE, window=WND_SIZE)
            # TODO: Initializing Siamese CBOW
            # TODO: Initializing Skip-Thoughts
            # TODO: Initializing ELMo

            initialize_models = False

        # Train Word2Vec
        model_W2V.train(sentences, epochs=EPOCHS)
        # TODO: Train Siamese CBOW
        # TODO: Train Skip-Thoughts
        # TODO: Train ELMo

# Testing Word2Vec
print(model_W2V.model.most_similar(positive=["happy"], topn=5))
# TODO: Testing Siamese CBOW
# TODO: Testing Skip-Thoughts
# TODO: Testing ELMo

# Saving Word2Vec
model_W2V.save(MODEL_DIR + "W2V" + MODEL_SUFFIX)
# TODO: Saving Siamese CBOW
# TODO: Saving Skip-Thoughts
# TODO: Saving ELMo