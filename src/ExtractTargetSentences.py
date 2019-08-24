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
import argparse
import numpy as np
import pandas as pd

import VNCPatternCounts
from Util import CorpusExtraction
from CForm import CForm
from SynLexFixedness import SynLexFixedness

# ------------- ARGS ------------- #
parser = argparse.ArgumentParser()

parser.add_argument("--TARG_DIR"         , "--candidates_directory"           , type=str, help="Location of the File Containing the Candidates in VNC-Token format.")
parser.add_argument("--SENT_DIR"         , "--output_sentences_dir"           , type=str, help="Location of the Output File Containing the Extracted Sentences.")
parser.add_argument("--SENT_LEMM_DIR"    , "--output_sentences_lemmatized_dir", type=str, help="Location of the Output File Containing the Lemmatized Extracted Sentences.")
parser.add_argument("--SENTVNC_DIR"      , "--output_vnc_dir"                 , type=str, help="Location of the Output File Containing the Extracted VNCs.")
parser.add_argument("--SENTVNC_LEMM_DIR" , "--output_vnc_lemmatized_dir"      , type=str, help="Location of the Output File Containing the Lemmatized Extracted VNCs.")
parser.add_argument("--CFORM_DIR"        , "--canonical_forms"                , type=str, help="Location of the File Indicating the Canonical Forms of the Candidates.")
parser.add_argument("--SYN_FIX_DIR"      , "--syntactical_fixedness"          , type=str, help="Location of the File Indicating the Syntactical Fixedness of the Candidates.")
parser.add_argument("--LEX_FIX_DIR"      , "--lexical_fixedness"              , type=str, help="Location of the File Indicating the Lexical Fixedness of the Candidates.")
parser.add_argument("--OVA_FIX_DIR"      , "--overall_fixedness"              , type=str, help="Location of the File Indicating the Overall Fixedness of the Candidates.")

parser.add_argument("--K"      , "--lexical_fixedness_k"    , type=int  , help="K Parameters for Number of Similar Verb/Nouns for Lexical Fixedness Calculation.")
parser.add_argument("--ALPHA"  , "--overall_fixedness_alpha", type=float, help="ALPHA Parameter for Overall Fixedness Calculation.")
parser.add_argument("--USE_LIN", help="Use flag to indicate Lin's Thesaurus for Similar Verb/Nouns. Defaults to Word2Vec and WordNet.", action="store_true")

parser.add_argument("--MAX_WINDOW", "--maximum_window", type=int, help="Maximum Window Size for Extracting Candidate VNCIs from Corpora.")

parser.add_argument("--NORM_FIX", "--normalize_fixedness_measures", help="Normalize Fixedness Measures Based on Absolute High Bounds.", action="store_true")

args = parser.parse_args()
# ------------- ARGS ------------- #

TARG_DIR         = "./targets/English_VNC_Cook/VNC-Tokens_cleaned"
SENT_DIR         = "./targets/Extracted_Sentences.txt"
SENT_LEMM_DIR    = "./targets/Extracted_Sentences_lemm.txt" 
SENTVNC_DIR      = "./targets/Extracted_Sentences_VNC.txt"
SENTVNC_LEMM_DIR = "./targets/Extracted_Sentences_VNC_lemm.txt" 
CFORM_DIR        = "./targets/CForms.csv"
SYN_FIX_DIR      = "./targets/SynFix.csv"
LEX_FIX_DIR      = "./targets/LexFix.csv"
OVA_FIX_DIR      = "./targets/OvaFix.csv"

# Other Parameters
CORPORA_PRE = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K"]
PAT_MODEL   = "./PatternCounts/PatternCounts_130619.pickle"
W2V_MODEL   = "./Word2Vec/models/W2V_ver1_lemm.model"
INDEXED_SUF = "_idx"
LEMMAS_SUF  = "_lemmas"
POSTAGS_SUF = "_posTags"
LOC_TOKEN   = "/"
VNC_TOKEN   = "_"

# Syntactical and Lexical Fixedness Parameters
K        = 50
ALPHA    = 0.6
LOG_BASE = 2
USE_LIN  = False

# VNC Patterns Parameters
MAX_WINDOW = 7

def main():
    # Initialize CForm Model
    cForm_model = CForm(modelDir=PAT_MODEL)

    # Initialize SynLexFixedness
    pat_SynLexFix = SynLexFixedness(modelDir=PAT_MODEL, w2vModelDir=W2V_MODEL, K=K)

    # Load VNC Tokens Dataset
    vncTokens = np.genfromtxt(TARG_DIR   , dtype='str', delimiter=' ')

    # ============= Original VNC Tokens Dataset ============= #

    # Load Corpora
    corpora = {}
    for prefix in CORPORA_PRE:
        print("Loading Corpora Sentences:", prefix)
        corpora[prefix] = CorpusExtraction.LoadCorpora(prefix, suffix=INDEXED_SUF)

    # Initialize Array
    sentences = []

    # Extract Sentences - VNC-Tokens
    with open(SENT_DIR, "w+") as sent_file:
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
    with open(SENT_LEMM_DIR, "w+") as sent_file:
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

    # ============= Original VNC Tokens Dataset ============= #
    # Initialize CForms Vector
    cForms = np.zeros(len(vncTokens))

    # Initilize Fixedness Vectors
    lexFix = np.zeros(len(vncTokens))
    synFix = np.zeros(len(vncTokens))
    ovaFix = np.zeros(len(vncTokens))

    # Determine CForms, get Fixedness Metrics, and extract VNC Pattern Subtext
    with open(SENTVNC_DIR, "w+") as sentvnc_file:
        with open(SENTVNC_LEMM_DIR, "w+") as sentvnc_lemm_file:
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

    if(NORM_FIX):
        synFix = synFix / np.max(np.abs(synFix))
        lexFix = lexFix / np.max(np.abs(lexFix))
        ovaFix = ovaFix / np.max(np.abs(ovaFix))

    np.savetxt(CFORM_DIR  , cForms, delimiter=",")
    np.savetxt(LEX_FIX_DIR, lexFix, delimiter=",")
    np.savetxt(SYN_FIX_DIR, synFix, delimiter=",")
    np.savetxt(OVA_FIX_DIR, ovaFix, delimiter=",")

if __name__ == '__main__':

    if(args.TARG_DIR):
        TARG_DIR = args.TARG_DIR
    if(args.SENT_DIR):
        SENT_DIR = args.SENT_DIR
    if(args.SENT_LEMM_DIR):
        SENT_LEMM_DIR = args.SENT_LEMM_DIR
    if(args.SENTVNC_DIR):
        SENTVNC_DIR = args.SENTVNC_DIR
    if(args.SENTVNC_LEMM_DIR):
        SENTVNC_LEMM_DIR = args.SENTVNC_LEMM_DIR
    if(args.CFORM_DIR):
        CFORM_DIR = args.CFORM_DIR
    if(args.SYN_FIX_DIR):
        SYN_FIX_DIR = args.SYN_FIX_DIR
    if(args.LEX_FIX_DIR):
        LEX_FIX_DIR = args.LEX_FIX_DIR
    if(args.OVA_FIX_DIR):
        OVA_FIX_DIR = args.OVA_FIX_DIR

    if(args.K):
        K = args.K
    if(args.ALPHA):
        ALPHA = args.ALPHA
    if(args.USE_LIN):
        USE_LIN = True
        W2V_MODEL = None

    if(args.MAX_WINDOW):
        MAX_WINDOW = args.MAX_WINDOW

    if(args.NORM_FIX):
        NORM_FIX  = True

    main()
