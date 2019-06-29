"""
    File:   GenerateWordEmbeddings
    Author: Jose Juan Zavala Iglesias
    Date:   29/05/2019

    Use the trained Embeddings Model (SkipThoughts) to generate word embeddings for target sentences.
"""

import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
from WordEmbeddings import Embeddings

SENT_DIR    = "../targets/Extracted_Sentences.txt"
SENTVNC_DIR = "../targets/Extracted_Sentences_VNC.txt"
EMBD_DIR    = "./embeddings.csv"
EMBDVNC_DIR = "./embeddings_VNC.csv"
MODEL_DIR   = ""

# Load Sentences
sentences = np.genfromtxt(SENT_DIR   , dtype='str', delimiter='\t')
sents_vnc = np.genfromtxt(SENTVNC_DIR, dtype='str', delimiter='\t')

# Set Sentences to Lowercase
for sent_id in range(len(sentences)):
    sentences[sent_id] = sentences[sent_id].lower()
    sents_vnc[sent_id] = sents_vnc[sent_id].lower()

# Load Model
model = Embeddings()

# Generate Embeddings
genEmbeddings    = model.GenerateFeatMatrix(sentences)
genEmbeddingsVNC = model.GenerateFeatMatrix(sents_vnc)

# Save Embeddings
np.savetxt(EMBD_DIR   , genEmbeddings   , delimiter=',')
np.savetxt(EMBDVNC_DIR, genEmbeddingsVNC, delimiter=',')
