"""
    File:   WordEmbeddings
    Author: Jose Juan Zavala Iglesias
    Date:   19/05/2019

    Word Embedding Models Wrapper for ease of use.
"""

import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
import wordEmbeddings as SiameseCBOW

PRETRAINNED_MODEL = "./models/cosine_sharedWeights_adadelta_lr_1_noGradClip_epochs_2_batch_100_neg_2_voc_65536x300_noReg_lc_noPreInit_vocab_65535.end_of_epoch_2.pickle"
VEC_SIZE = 300

class Embeddings:
    vec_dim = VEC_SIZE
    model   = None

    def __init__(self, pretrainned=PRETRAINNED_MODEL):
        self.model = SiameseCBOW.wordEmbeddings(pretrainned)

    def train(self, corpus, epochs=10):
        print("Not implemented")
        raise NotImplementedError

    def save(self, modelDir):
        print("Not implemented")
        raise NotImplementedError

    def load(self, pretrainned=PRETRAINNED_MODEL):
        self.model = SiameseCBOW.wordEmbeddings(pretrainned)

    def GetMostSimilar(self, word, topN=5):
        return self.model.most_similar(word, iTopN=topN)

    def GenerateFeatVector(self, sentence, vec_size = VEC_SIZE):
        if(self.model is None):
            print("Model not loading. Terminating Process.")
            return np.zeros((1, vec_size))
        
        featVector = self.model.getRandomEmbedding(sentence)

        return featVector

    def GenerateFeatMatrix(self, sentences, vec_size = VEC_SIZE):
        it = 0
        featMatrix = np.empty((sentences.shape[0], vec_size))

        for sentence in sentences:
            featMatrix[it, :] = self.GenerateFeatVector(sentence, vec_size=vec_size)
            it += 1

        return featMatrix

