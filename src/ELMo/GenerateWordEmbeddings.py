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
import argparse
import numpy as np
import pandas as pd
from WordEmbeddings import Embeddings

# ------------- ARGS ------------- #
parser = argparse.ArgumentParser()

parser.add_argument("--SENT_DIR"    , "--sentences_dir"       , type=str, help="Location of the File Containing the Sentences to be Transformed.")
parser.add_argument("--SENTVNC_DIR" , "--sentences_vncs_dir"  , type=str, help="Location of the File Containing the Sentences' VNCs to be Transformed.")
parser.add_argument("--EMBD_DIR"    , "--embeddings_dir"      , type=str, help="Location of the Output File with the Sentences Embeddings.")
parser.add_argument("--EMBDVNC_DIR" , "--embeddings_vncs_dir" , type=str, help="Location of the Output File with the Sentences' VNCs Embeddings.")

parser.add_argument("--MODEL_DIR"  , "--model_dir"  , type=str, help="Location of the Model to be Used for Embeddings.")
parser.add_argument("--BATCH_SIZE" , "--batch_size" , type=int, help="Number of Sentences to be Processed per Batch.")

args = parser.parse_args()
# ------------- ARGS ------------- #

# Tokens Directories
SENT_DIR    = "../targets/Extracted_Sentences.txt"
SENTVNC_DIR = "../targets/Extracted_Sentences_VNC.txt"
EMBD_DIR    = "./embeddings.csv"
EMBDVNC_DIR = "./embeddings_VNC.csv"

# Model Parameters
MODEL_DIR  = "https://tfhub.dev/google/elmo/2"
BATCH_SIZE = 300

# ======================================================= #

def main():
    print("Generating Embeddings:", EMBD_DIR, EMBDVNC_DIR)

    # Load Model
    model = Embeddings(hub_module=MODEL_DIR)

    # Load Sentences
    sentences = np.genfromtxt(SENT_DIR   , dtype='str', delimiter='\t')
    sents_vnc = np.genfromtxt(SENTVNC_DIR, dtype='str', delimiter='\t')

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
    np.savetxt(EMBD_DIR   , genEmbeddings   , delimiter=',')
    np.savetxt(EMBDVNC_DIR, genEmbeddingsVNC, delimiter=',')

# ======================================================= #
# ======================================================= #

if __name__ == '__main__':

    if(args.SENT_DIR):
        SENT_DIR = args.SENT_DIR
    if(args.SENTVNC_DIR):
        SENTVNC_DIR = args.SENTVNC_DIR
    if(args.EMBD_DIR):
        EMBD_DIR = args.EMBD_DIR
    if(args.EMBDVNC_DIR):
        EMBDVNC_DIR = args.EMBDVNC_DIR

    if(args.MODEL_DIR):
        MODEL_DIR = args.MODEL_DIR
    if(args.BATCH_SIZE):
        BATCH_SIZE = args.BATCH_SIZE

    main()
