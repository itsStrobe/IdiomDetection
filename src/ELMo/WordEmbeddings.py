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
import tensorflow as tf
import tensorflow_hub as tfhub

HUB_MODULE = "https://tfhub.dev/google/elmo/2"

ELMo_VEC_SIZE = 1024

class Embeddings:
    model = None

    def __init__(self, hub_module=HUB_MODULE):
        self.model = tfhub.Module(hub_module, trainable=True)

    def train(self, corpus, epochs=10):
        print("Not implemented")
        raise NotImplementedError

    def save(self, modelDir):
        print("Not implemented")
        raise NotImplementedError

    def load(self, hub_module=HUB_MODULE):
        self.model = tfhub.Module(hub_module, trainable=True)

    def GetMostSimilar(self, word, topN=5):
        print("Not implemented")
        raise NotImplementedError

    def GenerateFeatVector(self, sentence, vec_size = ELMo_VEC_SIZE):
        print("Not implemented. Use a single element list for single as parameter on GenerateFeatMatrix sentence embedding.")
        raise NotImplementedError

    def GenerateFeatMatrix(self, sentences, vec_size = ELMo_VEC_SIZE):
        featVector = self.model(sentences.tolist(), signature="default", as_dict=True)["elmo"]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            # return average of ELMo features
            return sess.run(tf.reduce_mean(featVector,1))
