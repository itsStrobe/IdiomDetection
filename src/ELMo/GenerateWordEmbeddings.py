"""
    File:   GenerateWordEmbeddings
    Author: Jose Juan Zavala Iglesias
    Date:   29/05/2019

    Use the trained Embeddings Model (ELMo) to generate word embeddings for target sentences.
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

# Tokens Directories
SENT_DIR    = ["../targets/Extracted_Sentences_lemm_cand.txt"]
SENTVNC_DIR = ["../targets/Extracted_Sentences_VNC_lemm_cand.txt"]
EMBD_DIR    = ["./embeddings_lemm_cand.csv"]
EMBDVNC_DIR = ["./embeddings_VNC_lemm_cand.csv"]

# Model Parameters
MODEL_DIR   = "https://tfhub.dev/google/elmo/2"
BATCH_SIZE  = 200

# Load Model
model = Embeddings(hub_module=MODEL_DIR)

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
    genEmbeddings    = np.zeros((sentences.shape[0], model.vec_dim))
    genEmbeddingsVNC = np.zeros((sents_vnc.shape[0], model.vec_dim))

    batch_count = 0
    num_sent    = sentences.shape[0]

    while batch_count*BATCH_SIZE < num_sent:
        print("Batch #", batch_count)
        start = batch_count*BATCH_SIZE
        end   = start + BATCH_SIZE
        if(end < num_sent):
            genEmbeddings[start : end]    = model.GenerateFeatMatrix(sentences[start : end])
            genEmbeddingsVNC[start : end] = model.GenerateFeatMatrix(sents_vnc[start : end])
        else:
            genEmbeddings[start : ]    = model.GenerateFeatMatrix(sentences[start : ])
            genEmbeddingsVNC[start : ] = model.GenerateFeatMatrix(sents_vnc[start : ])
        batch_count += 1

    print("Saving...")

    # Save Embeddings
    np.savetxt(embd_dir   , genEmbeddings   , delimiter=',')
    np.savetxt(embdvnc_dir, genEmbeddingsVNC, delimiter=',')

# ======================================================= #
