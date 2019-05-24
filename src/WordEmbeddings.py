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
import SiameseCBOW.wordEmbeddings as SiameseCBOW
from gensim.models import Word2Vec

WND_SIZE = 8
VEC_SIZE = 300

class Word2Vec_Embeddings:
    model = None

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

    def GenerateFeatMatrix(sentences, vec_size = VEC_SIZE):
        it = 0
        featMatrix = np.empty((sentences.shape[0], vec_size))

        for sentence in sentences:
            sentence = sentence.split()
            featMatrix[it, :] = self.GenerateFeatVector(sentence, vec_size=vec_size)
            it += 1

        return featMatrix

# TODO - Send to another file being executed with Python2. Wack.

class SiameseCBOW_Embeddings:
    model = None

    def __init__(self, pretrainned="./models/cosine_sharedWeights_adadelta_lr_1_noGradClip_epochs_2_batch_100_neg_2_voc_65536x300_noReg_lc_noPreInit_vocab_65535.end_of_epoch_2.pickle"):
        self.model = SiameseCBOW.wordEmbeddings(pretrainned)

    def train(self, corpus, epochs=10):
        print("Not implemented")
        raise NotImplementedError

    def save(self, modelDir):
        print("Not implemented")
        raise NotImplementedError

    def load(self, pretrainned="./models/cosine_sharedWeights_adadelta_lr_1_noGradClip_epochs_2_batch_100_neg_2_voc_65536x300_noReg_lc_noPreInit_vocab_65535.end_of_epoch_2.pickle"):
        self.model = SiameseCBOW.wordEmbeddings(pretrainned)

    def GetMostSimilar(self, word, topN=5):
        return self.model.most_similar(word, iTopN=topN)

    def GenerateFeatVector(self, sentence, vec_size = VEC_SIZE):
        if(self.model is None):
            print("Model not loading. Terminating Process.")
            return np.zeros(vec_size)
        
        featVector = np.empty((0, vec_size))

        for word in sentence:
            featVector = np.append(featVector, [self.model.getRandomEmbedding(word)], axis=0)

        return np.average(featVector, axis=0).reshape(1, vec_size)

    def GenerateFeatMatrix(sentences, vec_size = VEC_SIZE):
        it = 0
        featMatrix = np.empty((sentences.shape[0], vec_size))

        for sentence in sentences:
            sentence = sentence.split()
            featMatrix[it, :] = self.GenerateFeatVector(sentence, vec_size=vec_size)
            it += 1

        return featMatrix
