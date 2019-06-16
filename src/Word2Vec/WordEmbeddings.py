"""
    File:   WordEmbeddings
    Author: Jose Juan Zavala Iglesias
    Date:   19/05/2019

    Word Embedding Models Wrapper for ease of use
"""

import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

WND_SIZE = 8
VEC_SIZE = 300

class Embeddings:
    vec_dim = VEC_SIZE
    model   = None

    def __init__(self, corpus=None, size=VEC_SIZE, window=WND_SIZE, min_count=1, workers=4):
        if corpus == None:
            self.model = Word2Vec(size=size, window=window, min_count=min_count, workers=4)
        else:
            self.model = Word2Vec(corpus, size=size, window=window, min_count=min_count, workers=4)

    def train(self, corpus, epochs=10):
        self.model.train(corpus, total_examples=len(corpus), epochs=10)

    def save(self, modelDir):
        self.model.save(modelDir)

    def load(self, modelDir):
        self.model = Word2Vec.load(modelDir)

    def GetMostSimilar(self, word, topN=5):
        return self.model.most_similar(positive=[word], topn=topN)

    def GenerateFeatVector(self, sentence, vec_size = VEC_SIZE):
        if(self.model is None):
            print("Model not loading. Terminating Process.")
            return np.zeros(vec_size)
        
        featVector = np.empty((0, vec_size))

        for word in sentence:
            featVector = np.append(featVector, [self.model.wv[word]], axis=0)

        return np.average(featVector, axis=0).reshape(1, vec_size)

    def GenerateFeatMatrix(self, sentences, vec_size = VEC_SIZE):
        it = 0
        featMatrix = np.empty((sentences.shape[0], vec_size))

        for sentence in sentences:
            sentence = sentence.split()
            featMatrix[it, :] = self.GenerateFeatVector(sentence, vec_size=vec_size)
            it += 1

        return featMatrix
