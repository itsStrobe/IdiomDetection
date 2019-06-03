"""
    File:   ExtractTargetSentences
    Author: Jose Juan Zavala Iglesias
    Date:   29/02/2019

    Extract the target sentences for the VNC dataset.
"""

import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
from Util import CorpusExtraction

CORPORA_PRE  = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K"]
TARGETS_DIR = "./targets/English_VNC_Cook/VNC-Tokens_cleaned"
SENT_DIR    = "./targets/Extracted_Sentences.txt"
INDEXED_SUF = "_idx"
LOC_TOKEN   = "/"

# Load Corpora
corpora = {}
for prefix in CORPORA_PRE:
    print("Loading Corpora:", prefix)
    corpora[prefix] = CorpusExtraction.LoadCorpora(prefix, suffix=INDEXED_SUF)

# Load VNC Tokens Dataset
vncTokens = np.genfromtxt(TARGETS_DIR, dtype='str', delimiter=' ')

# Initialize Numpy Array
sentences = []

# Extract Sentences
with open(SENT_DIR, "w+") as file:    
    for sentenceLoc in vncTokens:
        print(sentenceLoc)
        fileLoc = sentenceLoc[2].split(LOC_TOKEN)
        sentNum = int(sentenceLoc[3])

        file.write(' '.join(corpora[fileLoc[0]][fileLoc[2]][sentNum]))
        file.write('\n')
