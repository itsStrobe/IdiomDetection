"""
    File:   GenerateWordEmbeddings
    Author: Jose Juan Zavala Iglesias
    Date:   29/02/2019

    Use the trained Embeddings Model (SkipThoughts) to generate word embeddings for target sentences.
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
MODEL_DIR = ""

# Load Sentences
sentences = np.genfromtxt(SENT_DIR, dtype='str', delimiter='\t')

# Load Model
model = Embeddings()

# Generate Embeddings
genEmbeddings = model.GenerateFeatMatrix(sentences)

# Save Embeddings
np.savetxt(EMBD_DIR, genEmbeddings, delimiter=',')
