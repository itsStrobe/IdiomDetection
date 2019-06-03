"""
    File:   TrainWordEmbeddings - Word2Vec
    Author: Jose Juan Zavala Iglesias
    Date:   21/02/2019

    Train Word Embedding Models with BNC XML Corpora and Store them for later use.
"""
import numpy as np
from Util import CorpusExtraction
from WordEmbeddings import Embeddings as W2V

CORPORA_PRE  = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K"]
MODEL_DIR    = "./models/"
MODEL_SUFFIX = "_ver1.model"

# Following values by King and Cook (2018)
VEC_SIZE = 300
WND_SIZE = 8
EPOCHS   = 5

model = None

CorporaIterator = [sent for sent in CorpusExtraction.IterateOverCorpora(CORPORA_PRE)]

## VOCABULARY INITIALIZATION ##
# Initializing Word2Vec's Vocabulary
print("Initializing Model")
model = W2V(corpus=CorporaIterator, size=VEC_SIZE, window=WND_SIZE)

## MODEL TRAINING ##
# Train Word2Vec
print("Training Model")
model.train(CorporaIterator, epochs=EPOCHS)

## MODEL TESTING ##
# Testing Word2Vec
print("Testing Model")
print(model.GetMostSimilar("happy"))

## SAVING MODELS ##
# Saving Word2Vec
print("Saving Model")
model.save(MODEL_DIR + "W2V" + MODEL_SUFFIX)
