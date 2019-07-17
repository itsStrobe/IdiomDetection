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

parser.add_argument("--MODEL_DIR" , "--model_dir" , type=str, help="Location of the Model to be Used for Embeddings.")

args = parser.parse_args()
# ------------- ARGS ------------- #

# Tokens Directories
SENT_DIR    = "../targets/Extracted_Sentences.txt"
SENTVNC_DIR = "../targets/Extracted_Sentences_VNC.txt"
EMBD_DIR    = "./embeddings.csv"
EMBDVNC_DIR = "./embeddings_VNC.csv"

# Model Parameters
MODEL_DIR = "./models/W2V_ver1.model"

# ======================================================= #

def main():
    print("Generating Embeddings:", EMBD_DIR, EMBDVNC_DIR)

    # Load Model
    model = Embeddings()
    model.load(MODEL_DIR)

    # Load Sentences
    sentences = np.genfromtxt(SENT_DIR   , dtype='str', delimiter='\t')
    sents_vnc = np.genfromtxt(SENTVNC_DIR, dtype='str', delimiter='\t')

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

    main()

