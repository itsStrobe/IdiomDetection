"""
    File:   GenerateWordEmbeddings
    Author: Jose Juan Zavala Iglesias
    Date:   29/05/2019

    Use the trained Embeddings Model (Word2Vec) to generate word embeddings for target sentences.
"""

import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
from WordEmbeddings import Embeddings

# Tokens Directories
SENT_DIR    = ["../targets/Extracted_Sentences.txt"    , "../targets/Extracted_Sentences_cand.txt"    ]
SENTVNC_DIR = ["../targets/Extracted_Sentences_VNC.txt", "../targets/Extracted_Sentences_VNC_cand.txt"]
EMBD_DIR    = ["./embeddings.csv"                      , "./embeddings_cand.csv"                      ]
EMBDVNC_DIR = ["./embeddings_VNC.csv"                  , "./embeddings_VNC_cand.csv"                  ]

# Tokens Directories - Lemmas
SENT_LEMM_DIR   = ["../targets/Extracted_Sentences_lemm.txt"    , "../targets/Extracted_Sentences_lemm_cand.txt"]
SENTVNC_LEM_DIR = ["../targets/Extracted_Sentences_VNC_lemm.txt", "../targets/Extracted_Sentences_VNC_lemm_cand.txt"]
EMBD_LEM_DIR    = ["./embeddings_lemm.csv"                      , "./embeddings_lemm_cand.csv"]
EMBDVNC_LEM_DIR = ["./embeddings_VNC_lemm.csv"                  , "./embeddings_VNC_lemm_cand.csv"]

# Model Parameters
MODEL_DIR      = "./models/W2V_ver1.model"
MODEL_LEMM_DIR = "./models/W2V_ver1_lemm.model"

# Load Model
model = Embeddings()
model.load(MODEL_DIR)
model_lemm = Embeddings()
model_lemm.load(MODEL_LEMM_DIR)

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
# ======================================================= #

for sent_dir, sentvnc_dir, embd_dir, embdvnc_dir in zip(SENT_LEMM_DIR, SENTVNC_LEM_DIR, EMBD_LEM_DIR, EMBDVNC_LEM_DIR):
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
    genEmbeddings    = model_lemm.GenerateFeatMatrix(sentences)
    genEmbeddingsVNC = model_lemm.GenerateFeatMatrix(sents_vnc)

    print("Saving...")

    # Save Embeddings
    np.savetxt(embd_dir   , genEmbeddings   , delimiter=',')
    np.savetxt(embdvnc_dir, genEmbeddingsVNC, delimiter=',')

# ======================================================= #

