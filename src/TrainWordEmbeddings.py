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

model_W2V = None
model_SCB = None
model_SkT = None
model_ELM = None

CorporaIterator = [sent for sent in CorpusExtraction.IterateOverCorpora(CORPORA_PRE)]


## VOCABULARY INITIALIZATION ##
# Initializing Word2Vec's Vocabulary
model_W2V = W2V(corpus=CorporaIterator, size=VEC_SIZE, window=WND_SIZE)
# TODO: Initializing Siamese CBOW's Vocabulary
# TODO: Initializing Skip-Thoughts' Vocabulary
# TODO: Initializing ELMo's Vocabulary

## MODEL TRAINING ##
# Train Word2Vec
model_W2V.train(CorporaIterator, epochs=EPOCHS)
# TODO: Train Siamese CBOW
# TODO: Train Skip-Thoughts
# TODO: Train ELMo

## MODEL TESTING ##
# Testing Word2Vec
print(model_W2V.model.most_similar(positive=["happy"], topn=5))
# TODO: Testing Siamese CBOW
# TODO: Testing Skip-Thoughts
# TODO: Testing ELMo

## SAVING MODELS ##
# Saving Word2Vec
model_W2V.save(MODEL_DIR + "W2V" + MODEL_SUFFIX)
# TODO: Saving Siamese CBOW
# TODO: Saving Skip-Thoughts
# TODO: Saving ELMo