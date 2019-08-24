"""
    File:   Experiment
    Author: Jose Juan Zavala Iglesias
    Date:   05/07/2019

    Unsupervised (Cosine Similarity) Classification evaluation for VNC Tokens Dataset using word embeddings generated with:
        >Word2Vec
        >Skip-Thoughts
        >Siamese CBOW
        >ELMo

    Added Fazly et. al (2009) Metrics
"""

import os
import re
import sys
import pickle
import argparse
import numpy as np
import pandas as pd

# ------------- ARGS ------------- #
parser = argparse.ArgumentParser()

parser.add_argument("--DATASET_DIR"   , "--dataset_directory"         , type=str, help="Location of the File Containing the Dataset in VNC-Token format.")
parser.add_argument("--BEST_EMBD_DIR" , "--best_embeddings_directory" , type=str, help="Location of the File Containing the Embeddings that Correspond to the Defined Dataset.")
parser.add_argument("--CFORM_DIR"     , "--canonical_forms"           , type=str, help="Location of the File Indicating the Canonical Forms of the Candidates.")
parser.add_argument("--SYN_FIX_DIR"   , "--syntactical_fixedness"     , type=str, help="Location of the File Indicating the Syntactical Fixedness of the Candidates.")
parser.add_argument("--LEX_FIX_DIR"   , "--lexical_fixedness"         , type=str, help="Location of the File Indicating the Lexical Fixedness of the Candidates.")
parser.add_argument("--OVA_FIX_DIR"   , "--overall_fixedness"         , type=str, help="Location of the File Indicating the Overall Fixedness of the Candidates.")

parser.add_argument("--BEST_SVM" , "--best_svm_model_directory" , type=str, help="Best SVM Model Location")

parser.add_argument("--USE_CFORM"   , help="Use flag to indicate if CForm Feature Should Be Used for the SVM Classifier."               , action="store_true")
parser.add_argument("--USE_SYN_FIX" , help="Use flag to indicate if Syntactic Fixedness Feature Should Be Used for the SVM Classifier." , action="store_true")
parser.add_argument("--USE_LEX_FIX" , help="Use flag to indicate if Lexical Fixedness Feature Should Be Used for the SVM Classifier."   , action="store_true")
parser.add_argument("--USE_OVA_FIX" , help="Use flag to indicate if Overall Fixedness Feature Should Be Used for the SVM Classifier."   , action="store_true")

args = parser.parse_args()
# ------------- ARGS ------------- #

# Files and Directories:
DATASET_DIR   = "./targets/VNC-Tokens_candidates"
BEST_EMBD_DIR = "./Word2Vec/embeddings_cand.csv"
CFORM_DIR     = "./targets/CForms_cand.csv"
SYN_FIX_DIR   = "./targets/SynFix_cand.csv"
LEX_FIX_DIR   = "./targets/LexFix_cand.csv"
OVA_FIX_DIR   = "./targets/OvaFix_cand.csv"

# SVM Classifier Directory:
BEST_SVM   = "./SVM_Models/W2V_clean.model"

# SVM Classifiers Features:
USE_CFORM   = False
USE_SYN_FIX = False
USE_LEX_FIX = False
USE_OVA_FIX = False

def main():

    # -- EXTRACT DATASETS -- #
    df_vncs = pd.read_csv(DATASET_DIR, header=None, sep=' ')
    targets = df_vncs[0].values.flatten()

    # Sentence Embeddings
    features = np.genfromtxt(BEST_EMBD_DIR, delimiter=',')

    # -- ADD FAZLY's METRICS -- #
    if(USE_CFORM):
        cForms = np.genfromtxt(CFORM_DIR, delimiter=',')
        cForms = cForms.reshape((cForms.size, 1))

        features = np.append(features, cForms, axis=1)

    if(USE_SYN_FIX):
        synFix = np.genfromtxt(SYN_FIX_DIR, delimiter=',')
        synFix = synFix.reshape((synFix.size, 1))

        features = np.append(features, synFix, axis=1)

    if(USE_LEX_FIX):
        lexFix = np.genfromtxt(LEX_FIX_DIR, delimiter=',')
        lexFix = lexFix.reshape((lexFix.size, 1))

        features = np.append(features, lexFix, axis=1)

    if(USE_OVA_FIX):
        ovaFix = np.genfromtxt(OVA_FIX_DIR, delimiter=',')
        ovaFix = ovaFix.reshape((ovaFix.size, 1))

        features = np.append(features, ovaFix, axis=1)

    # - Get SVM Predictions
    if(os.path.isfile(BEST_SVM)):
        with open(BEST_SVM, 'rb') as svmFile:
            svm_clf = pickle.load(svmFile)
    else:
        print("File does not exists:", BEST_SVM)
        return
    silv_std = svm_clf.predict(features)

    for idx in range(len(silv_std)):
        if(silv_std[idx] == True):
            targets[idx] = 'I'
        else:
            targets[idx] = 'L'

    df_vncs[0] = targets
    df_vncs.to_csv(path_or_buf=DATASET_DIR, sep=' ', header=False, index=False)


if __name__ == '__main__':

    if(args.DATASET_DIR):
        DATASET_DIR = args.DATASET_DIR
    if(args.BEST_EMBD_DIR):
        BEST_EMBD_DIR = args.BEST_EMBD_DIR
    if(args.CFORM_DIR):
        CFORM_DIR = args.CFORM_DIR
    if(args.SYN_FIX_DIR):
        SYN_FIX_DIR = args.SYN_FIX_DIR
    if(args.LEX_FIX_DIR):
        LEX_FIX_DIR = args.LEX_FIX_DIR
    if(args.OVA_FIX_DIR):
        OVA_FIX_DIR = args.OVA_FIX_DIR

    if(args.BEST_SVM):
        BEST_SVM = args.BEST_SVM

    if(args.USE_CFORM):
        USE_CFORM = True
    if(args.USE_SYN_FIX):
        USE_SYN_FIX = True
    if(args.USE_LEX_FIX):
        USE_LEX_FIX = True
    if(args.USE_OVA_FIX):
        USE_OVA_FIX = True

    main()
