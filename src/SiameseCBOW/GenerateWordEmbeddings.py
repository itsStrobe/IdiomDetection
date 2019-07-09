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
from WordEmbeddings import Embeddings

# Tokens Directories
SENT_DIR    = ["../targets/Extracted_Sentences.txt"    , "../targets/Extracted_Sentences_lemm.txt"    , "../targets/Extracted_Sentences_cand.txt"    , "../targets/Extracted_Sentences_lemm_cand.txt"]
SENTVNC_DIR = ["../targets/Extracted_Sentences_VNC.txt", "../targets/Extracted_Sentences_VNC_lemm.txt", "../targets/Extracted_Sentences_VNC_cand.txt", "../targets/Extracted_Sentences_VNC_lemm_cand.txt"]
EMBD_DIR    = ["./embeddings.csv"                      , "./embeddings_lemm.csv"                      , "./embeddings_cand.csv"                      , "./embeddings_lemm_cand.csv"]
EMBDVNC_DIR = ["./embeddings_VNC.csv"                  , "./embeddings_VNC_lemm.csv"                  , "./embeddings_VNC_cand.csv"                  , "./embeddings_VNC_lemm_cand.csv"]

# Model Parameters
MODEL_DIR   = "./models/cosine_sharedWeights_adadelta_lr_1_noGradClip_epochs_2_batch_100_neg_2_voc_65536x300_noReg_lc_noPreInit_vocab_65535.end_of_epoch_2.pickle"

# Load Model
model = Embeddings(pretrainned=MODEL_DIR)

# ======================================================= #

for sent_dir, sentvnc_dir, embd_dir, embdvnc_dir in zip(SENT_DIR, SENTVNC_DIR, EMBD_DIR, EMBDVNC_DIR):
    print("Generating Embeddings:", embd_dir, embdvnc_dir)

    # Load Sentences
    sentences = np.genfromtxt(sent_dir   , dtype='str', delimiter='\t')
    sents_vnc = np.genfromtxt(sentvnc_dir, dtype='str', delimiter='\t')

    # Set Sentences to Lowercase
    for sent_id in range(len(sentences)):
        sentences[sent_id] = sentences[sent_id].lower()
        sents_vnc[sent_id] = sents_vnc[sent_id].lower()

    print("Generating Embeddings...")

    # Generate Embeddings
    genEmbeddings    = model.GenerateFeatMatrix(sentences)
    genEmbeddingsVNC = model.GenerateFeatMatrix(sents_vnc)

    print("Saving...")

    # Save Embeddings
    np.savetxt(embd_dir   , genEmbeddings   , delimiter=',')
    np.savetxt(embdvnc_dir, genEmbeddingsVNC, delimiter=',')

# ======================================================= #
