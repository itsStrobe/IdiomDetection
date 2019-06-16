"""
    File:   GenerateWordEmbeddings
    Author: Jose Juan Zavala Iglesias
    Date:   29/05/2019

    Use the trained Embeddings Model (SiameseCBOW) to generate word embeddings for target sentences.
"""

import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
from Util import CorpusExtraction
from WordEmbeddings import Embeddings

SENT_DIR  = "../targets/Extracted_Sentences.txt"
EMBD_DIR  = "./embeddings.csv"
MODEL_DIR = "./models/cosine_sharedWeights_adadelta_lr_1_noGradClip_epochs_2_batch_100_neg_2_voc_65536x300_noReg_lc_noPreInit_vocab_65535.end_of_epoch_2.pickle"

# Load Sentences
sentences = np.genfromtxt(SENT_DIR, dtype='str', delimiter='\t')

# Load Model
model = Embeddings(pretrainned=MODEL_DIR)

# Generate Embeddings
genEmbeddings = model.GenerateFeatMatrix(sentences)

# Save Embeddings
np.savetxt(EMBD_DIR, genEmbeddings, delimiter=',')
