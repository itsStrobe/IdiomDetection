"""
    File:   ExtractTargetSentences
    Author: Jose Juan Zavala Iglesias
    Date:   01/07/2019

    Extract the target sentences for the VNC Candidates Dataset.
"""

import os
import re
import sys
import pickle
import numpy as np
import pandas as pd

import VNCPatternCounts
from Util import CorpusExtraction
from CForm import CForm
from SynLexFixedness import SynLexFixedness

TARG_CD_DIR         = "./targets/VNC-Tokens_candidates"
SENT_CD_DIR         = "./targets/Extracted_Sentences_cand.txt"
SENT_LEMM_CD_DIR    = "./targets/Extracted_Sentences_lemm_cand.txt"
SENTVNC_CD_DIR      = "./targets/Extracted_Sentences_VNC_cand.txt"
SENTVNC_LEMM_CD_DIR = "./targets/Extracted_Sentences_VNC_lemm_cand.txt" 
CFORM_CD_DIR        = "./targets/CForms_cand.csv"
SYN_FIX_CD_DIR      = "./targets/SynFix_cand.csv"
LEX_FIX_CD_DIR      = "./targets/LexFix_cand.csv"
OVA_FIX_CD_DIR      = "./targets/OvaFix_cand.csv"

# Other Parameters
CORPORA_PRE = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K"]
PAT_MODEL   = "./PatternCounts/PatternCounts_130619.pickle"
W2V_MODEL   = None # "./Word2Vec/models/W2V_ver1.model"
INDEXED_SUF = "_idx"
LEMMAS_SUF  = "_lemmas"
POSTAGS_SUF = "_posTags"
LOC_TOKEN   = "/"
VNC_TOKEN   = "_"

# Syntactical and Lexical Fixedness Parameters
K        = 50
ALPHA    = 0.6
LOG_BASE = 2
USE_LIN  = True

# VNC Patterns Parameters
MAX_WINDOW = 7

# Initialize CForm Model
cForm_model = CForm(modelDir=PAT_MODEL)

# Initialize SynLexFixedness
pat_SynLexFix = SynLexFixedness(modelDir=PAT_MODEL, w2vModelDir=W2V_MODEL, K=K)

# Load VNC Tokens Dataset
vncTokens = np.genfromtxt(TARG_CD_DIR, dtype='str', delimiter=' ')

# ============ Candidates VNC Tokens Dataset ============ #

# Load Corpora
corpora = {}
for prefix in CORPORA_PRE:
    print("Loading Corpora Sentences:", prefix)
    corpora[prefix] = CorpusExtraction.LoadCorpora(prefix, suffix=INDEXED_SUF)

# Initialize Array
sentences = []

# Extract Sentences - VNC-Candidates
with open(SENT_CD_DIR, "w+") as sent_file:
    for sentenceLoc in vncTokens:
        fileLoc = sentenceLoc[2].split(LOC_TOKEN)
        sentNum = int(sentenceLoc[3])

        sent = corpora[fileLoc[0]][fileLoc[2]][sentNum]

        sentences.append(sent)

        sent_file.write(' '.join(sent))
        sent_file.write('\n')

# Free Memory
corpora = None

# ======================================================= #

# Load Lemmatized Corpora
corpora_lemm = {}
for prefix in CORPORA_PRE:
    print("Loading Corpora Sentences:", prefix)
    corpora_lemm[prefix] = CorpusExtraction.LoadCorpora(prefix, suffix=LEMMAS_SUF)

# Initialize Lemmatized Array
sentences_lemm = []

# Extract Sentences - VNC-Tokens
with open(SENT_LEMM_CD_DIR, "w+") as sent_file:
    for sentenceLoc in vncTokens:
        fileLoc = sentenceLoc[2].split(LOC_TOKEN)
        sentNum = int(sentenceLoc[3])

        sent = corpora_lemm[fileLoc[0]][fileLoc[2]][sentNum]

        sentences_lemm.append(sent)

        sent_file.write(' '.join(sent))
        sent_file.write('\n')

# Free Memory
corpora_lemm = None

# ======================================================= #
# ======================================================= #

corpora_tags = {}
for prefix in CORPORA_PRE:
    print("Loading Corpora Pos Tags:", prefix)
    corpora_tags[prefix] = CorpusExtraction.LoadCorpora(prefix, suffix=POSTAGS_SUF)

# ======================================================= #
# ======================================================= #

# ============ Candidates VNC Tokens Dataset ============ #
# Initialize CForms Vector
cForms = np.zeros(len(vncTokens))

# Initilize Fixedness Vectors
lexFix = np.zeros(len(vncTokens))
synFix = np.zeros(len(vncTokens))
ovaFix = np.zeros(len(vncTokens))

# Determine CForms, get Fixedness Metrics, and extract VNC Pattern Subtext
with open(SENTVNC_CD_DIR, "w+") as sentvnc_file:
    with open(SENTVNC_LEMM_CD_DIR, "w+") as sentvnc_lemm_file:
        it = 0
        for sentenceLoc, sent, sent_lemm in zip(vncTokens, sentences, sentences_lemm):
            fileLoc = sentenceLoc[2].split(LOC_TOKEN)
            sentNum = int(sentenceLoc[3])

            vnc     = sentenceLoc[1].split(VNC_TOKEN)
            posTags = corpora_tags[fileLoc[0]][fileLoc[2]][sentNum]

            if(cForm_model.IsCForm(vnc[0], vnc[1], sent, posTags)):
                cForms[it] = 1

            lexFix[it] = pat_SynLexFix.Fixedness_Lex(vnc[0], vnc[1], vK=K, nK=K, logBase=LOG_BASE, useLin=USE_LIN)
            synFix[it] = pat_SynLexFix.Fixedness_Syn(vnc[0], vnc[1], logBase=LOG_BASE)
            ovaFix[it] = pat_SynLexFix.Fixedness_Overall(vnc[0], vnc[1], alpha=ALPHA, vK=K, nK=K, logBase=LOG_BASE, useLin=USE_LIN)

            sentvnc_idx = VNCPatternCounts.ExtractPatternRangeFromSentence(sent_lemm, posTags, vnc, max_window=MAX_WINDOW)

            sentvnc = sent[sentvnc_idx[0]:sentvnc_idx[1] + 1]
            sentvnc_file.write(' '.join(sentvnc))
            sentvnc_file.write('\n')

            sentvnc_lemm = sent_lemm[sentvnc_idx[0]:sentvnc_idx[1] + 1]
            sentvnc_lemm_file.write(' '.join(sentvnc_lemm))
            sentvnc_lemm_file.write('\n')

            it += 1

# ovaFix = ovaFix / np.max(ovaFix)

np.savetxt(CFORM_CD_DIR , cForms, delimiter=",")
np.savetxt(SYN_FIX_CD_DIR, lexFix, delimiter=",")
np.savetxt(LEX_FIX_CD_DIR, synFix, delimiter=",")
np.savetxt(OVA_FIX_CD_DIR, ovaFix, delimiter=",")
