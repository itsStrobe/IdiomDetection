"""
    File:   WordEmbeddings
    Author: Jose Juan Zavala Iglesias
    Date:   19/02/2019

    Word Embedding Models Wrapper for ease of use
"""

import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
import skipthoughts as SkipThoughts

WND_SIZE = 8
VEC_SIZE = 300

class Embeddings:
    model   = None
    encoder = None

    def __init__(self):
        self.model   = SkipThoughts.load_model()
        self.encoder = SkipThoughts.Encoder(self.model)

    def train(self, corpus, epochs=10):
        print("Not implemented")
        raise NotImplementedError

    def save(self, modelDir):
        print("Not implemented")
        raise NotImplementedError

    def load(self):
        self.model   = SkipThoughts.load_model()
        self.encoder = SkipThoughts.Encoder(model)

    def GetMostSimilar(self, word, topN=5):
        print("Not implemented")
        raise NotImplementedError

    def GenerateFeatVector(self, sentence, vec_size = VEC_SIZE):
        print("Not implemented. Use a single element list for single as parameter on GenerateFeatMatrix sentence embedding.")
        raise NotImplementedError

    def GenerateFeatMatrix(self, sentences, vec_size = VEC_SIZE):
        it = 0
        featMatrix = np.empty((sentences.shape[0], vec_size))

        featMatrix = self.encoder.encode(sentences)

        return featMatrix
