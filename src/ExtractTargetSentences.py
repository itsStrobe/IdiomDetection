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
from CForm import CForm

CORPORA_PRE  = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K"]
TARGETS_DIR = "./targets/English_VNC_Cook/VNC-Tokens_cleaned"
SENT_DIR    = "./targets/Extracted_Sentences.txt" 
CFORM_DIR   = "./targets/CForms.csv"
CFORM_MODEL = "./PatternCounts/PatternCounts_100619.pickle"
INDEXED_SUF = "_idx"
POSTAGS_SUF = "_posTags"
LOC_TOKEN   = "/"
VNC_TOKEN   = "_"

# Load Corpora
corpora      = {}
corpora_tags = {}
for prefix in CORPORA_PRE:
    print("Loading Corpora Sentences:", prefix)
    corpora[prefix] = CorpusExtraction.LoadCorpora(prefix, suffix=INDEXED_SUF)

# Load VNC Tokens Dataset
vncTokens = np.genfromtxt(TARGETS_DIR, dtype='str', delimiter=' ')

# Initialize Array
sentences = []

# Extract Sentences
with open(SENT_DIR, "w+") as sent_file:
    for sentenceLoc in vncTokens:
        print(sentenceLoc)
        fileLoc = sentenceLoc[2].split(LOC_TOKEN)
        sentNum = int(sentenceLoc[3])

        sent = corpora[fileLoc[0]][fileLoc[2]][sentNum]

        sentences.append(sent)

        sent_file.write(' '.join(sent))
        sent_file.write('\n')

# ===================================================== #

# Free Memory
corpora = None

for prefix in CORPORA_PRE:
    print("Loading Corpora Pos Tags:", prefix)
    corpora_tags[prefix] = CorpusExtraction.LoadCorpora(prefix, suffix=POSTAGS_SUF)

# Initialize CForm Model
cForm_model = CForm(modelDir=CFORM_MODEL)

# Initialize CForms Vector
cForms = np.zeros(len(vncTokens))

# Determine CForms
it = 0
for sentenceLoc, sent in zip(vncTokens, sentences):
    print(sentenceLoc)
    fileLoc = sentenceLoc[2].split(LOC_TOKEN)
    sentNum = int(sentenceLoc[3])

    vnc     = sentenceLoc[1].split(VNC_TOKEN)
    posTags = corpora_tags[fileLoc[0]][fileLoc[2]][sentNum]

    if(cForm_model.IsCForm(vnc[0], vnc[1], sent, posTags)):
        cForms[it] = 1

    it += 1

np.savetxt(CFORM_DIR, cForms, delimiter=",")
