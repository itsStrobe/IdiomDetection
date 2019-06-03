"""
    File:   GenerateWordEmbeddings
    Author: Jose Juan Zavala Iglesias
    Date:   29/02/2019

    Use the trained Embeddings Model (ELMo) to generate word embeddings for target sentences.
"""

import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
from Util import CorpusExtraction
from WordEmbeddings import Embeddings

SENT_DIR   = "../targets/Extracted_Sentences.txt"
EMBD_DIR   = "./embeddings.csv"
MODEL_DIR  = "https://tfhub.dev/google/elmo/2"
BATCH_SIZE = 400

# Load Sentences
sentences = np.genfromtxt(SENT_DIR, dtype='str', delimiter='\t')

# Load Model
model = Embeddings(hub_module=MODEL_DIR)

# Generate Embeddings
genEmbeddings = np.zeros((sentences.shape[0], model.vec_dim))
batch_count = 0
num_sent    = sentences.shape[0]
while batch_count*BATCH_SIZE < num_sent:
    print("Batch #", batch_count)
    start = batch_count*BATCH_SIZE
    end   = start + BATCH_SIZE
    if(end < num_sent):
        genEmbeddings[start : end] = model.GenerateFeatMatrix(sentences[start : end])
    else:
        genEmbeddings[start : ]    = model.GenerateFeatMatrix(sentences[start : ])
    batch_count += 1

# for sentence in sentences:
#     print(sentence)
#     genEmbeddings[count] = model.GenerateFeatMatrix(np.array([sentence]))
#     count += 1

# Save Embeddings
np.savetxt(EMBD_DIR, genEmbeddings, delimiter=',')
