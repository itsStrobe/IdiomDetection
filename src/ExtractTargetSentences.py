"""
    File:   ExtractTargetSentences
    Author: Jose Juan Zavala Iglesias
    Date:   29/05/2019

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
from SynLexFixedness import SynLexFixedness

CORPORA_PRE  = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K"]
TARGETS_DIR = "./targets/English_VNC_Cook/VNC-Tokens_cleaned"
SENT_DIR    = "./targets/Extracted_Sentences.txt" 
CFORM_DIR   = "./targets/CForms.csv"
SYN_FIX_DIR = "./targets/SynFix.csv"
LEX_FIX_DIR = "./targets/LexFix.csv"
OVA_FIX_DIR = "./targets/OvaFix.csv"
PAT_MODEL   = "./PatternCounts/PatternCounts_130619.pickle"
W2V_MODEL   = None # "./Word2Vec/models/W2V_ver1.model"
INDEXED_SUF = "_idx"
POSTAGS_SUF = "_posTags"
LOC_TOKEN   = "/"
VNC_TOKEN   = "_"

# Syntactical and Lexical Fixedness Parameters
K        = 50
ALPHA    = 0.6
LOG_BASE = 2
USE_LIN  = True

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
cForm_model = CForm(modelDir=PAT_MODEL)

# Initialize SynLexFixedness
pat_SynLexFix = SynLexFixedness(modelDir=PAT_MODEL, w2vModelDir=W2V_MODEL, K=K)

# Initialize CForms Vector
cForms = np.zeros(len(vncTokens))

# Initilize Fixedness Vectors
lexFix = np.zeros(len(vncTokens))
synFix = np.zeros(len(vncTokens))
ovaFix = np.zeros(len(vncTokens))

# Determine CForms
it = 0
for sentenceLoc, sent in zip(vncTokens, sentences):
    fileLoc = sentenceLoc[2].split(LOC_TOKEN)
    sentNum = int(sentenceLoc[3])

    vnc     = sentenceLoc[1].split(VNC_TOKEN)
    posTags = corpora_tags[fileLoc[0]][fileLoc[2]][sentNum]

    if(cForm_model.IsCForm(vnc[0], vnc[1], sent, posTags)):
        cForms[it] = 1

    lexFix[it] = pat_SynLexFix.Fixedness_Lex(vnc[0], vnc[1], vK=K, nK=K, logBase=LOG_BASE, useLin=USE_LIN)
    synFix[it] = pat_SynLexFix.Fixedness_Syn(vnc[0], vnc[1], logBase=LOG_BASE)
    ovaFix[it] = pat_SynLexFix.Fixedness_Overall(vnc[0], vnc[1], alpha=ALPHA, vK=K, nK=K, logBase=LOG_BASE, useLin=USE_LIN)

    it += 1

np.savetxt(CFORM_DIR  , cForms, delimiter=",")
np.savetxt(LEX_FIX_DIR, lexFix, delimiter=",")
np.savetxt(SYN_FIX_DIR, synFix, delimiter=",")
np.savetxt(OVA_FIX_DIR, ovaFix, delimiter=",")
