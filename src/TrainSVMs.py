"""
    File:   TrainSVMs
    Author: Jose Juan Zavala Iglesias
    Date:   10/07/2019

    SVM Training using word embeddings generated with:
        >Word2Vec
        >Skip-Thoughts
        >Siamese CBOW
        >ELMo

    Added Fazly's Metrics
"""

import os
import re
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.utils import shuffle

# ------------- ARGS ------------- #
parser = argparse.ArgumentParser()

parser.add_argument("--TARG_DIR"      , "--targets_directory"                 , type=str, help="Location of the File Containing the Targets in VNC-Token format.")
parser.add_argument("--W2V_DIR"       , "--w2v_directory"                     , type=str, help="Location of the Input Dir Containing the Word2Vec Embeddings.")
parser.add_argument("--SCBOW_DIR"     , "--scbow_directory"                   , type=str, help="Location of the Input Dir Containing the Siamese SCBOW Embeddings.")
parser.add_argument("--SKIP_DIR"      , "--skip-thoughts_directory"           , type=str, help="Location of the Input Dir Containing the Skip-Thoughts Embeddings.")
parser.add_argument("--ELMO_DIR"      , "--elmo_directory"                    , type=str, help="Location of the Input Dir Containing the ELMo Embeddings.")
parser.add_argument("--CFORM_DIR"     , "--canonical_forms"                   , type=str, help="Location of the File Indicating the Canonical Forms of the Candidates.")
parser.add_argument("--SYN_FIX_DIR"   , "--syntactical_fixedness"             , type=str, help="Location of the File Indicating the Syntactical Fixedness of the Candidates.")
parser.add_argument("--LEX_FIX_DIR"   , "--lexical_fixedness"                 , type=str, help="Location of the File Indicating the Lexical Fixedness of the Candidates.")
parser.add_argument("--OVA_FIX_DIR"   , "--overall_fixedness"                 , type=str, help="Location of the File Indicating the Overall Fixedness of the Candidates.")
parser.add_argument("--VECTORS_FILE"  , "--embedded_vectors_file"             , type=str, help="Name of the Embeddings File.")
parser.add_argument("--W2V_RESULTS"   , "--w2v_results_file_prefix"           , type=str, help="Location of the Output File Containing the Cross-Validation Results for Word2Vec's SVM.")
parser.add_argument("--SCBOW_RESULTS" , "--scbow_results_file_prefix"         , type=str, help="Location of the Output File Containing the Cross-Validation Results for Siamese CBOW's SVM.")
parser.add_argument("--SKIP_RESULTS"  , "--skip-thoughts_results_file_prefix" , type=str, help="Location of the Output File Containing the Cross-Validation Results for Skip-Thoughts's SVM.")
parser.add_argument("--ELMO_RESULTS"  , "--elmo_results_file_prefix"          , type=str, help="Location of the Output File Containing the Cross-Validation Results for ELMo's SVM.")
parser.add_argument("--FILE_EXT"      , "--output_file_extension"             , type=str, help="File Extension for Output Files.")

parser.add_argument("--MODELS_DIR" , "--models_directory" , type=str, help="Models Directory.")
parser.add_argument("--MODEL_EXT"  , "--model_extension"  , type=str, help="Model Name Extension.")

parser.add_argument("--USE_CFORM"   , help="Use flag to indicate if CForm Feature Should Be Added."               , action="store_true")
parser.add_argument("--USE_SYN_FIX" , help="Use flag to indicate if Syntactic Fixedness Feature Should Be Added." , action="store_true")
parser.add_argument("--USE_LEX_FIX" , help="Use flag to indicate if Lexical Fixedness Feature Should Be Added."   , action="store_true")
parser.add_argument("--USE_OVA_FIX" , help="Use flag to indicate if Overall Fixedness Feature Should Be Added."   , action="store_true")

parser.add_argument("--C_W2V"   , "--W2V_C_Parameter"           , type=float, help="SVM C Parameter for Word2Vec Embeddings.")
parser.add_argument("--C_SKIP"  , "--Skip-Thoughts_C_Parameter" , type=float, help="SVM C Parameter for Skip-Thoughts Embeddings.")
parser.add_argument("--C_SCBOW" , "--CBOW_C_Parameter"          , type=float, help="SVM C Parameter for Siamese CBOW Embeddings.")
parser.add_argument("--C_ELMo"  , "--ELMO_C_Parameter"          , type=float, help="SVM C Parameter for ELMo Embeddings.")
parser.add_argument("--KERNEL"  , "--kernel_type"               , type=str  , help="SVM Kernel.")

args = parser.parse_args()
# ------------- ARGS ------------- #

# Files and Directories:
TARG_DIR      = "./targets/English_VNC_Cook/VNC-Tokens_cleaned"
W2V_DIR       = "./Word2Vec/"
SCBOW_DIR     = "./SiameseCBOW/"
SKIP_DIR      = "./SkipThoughts/"
ELMO_DIR      = "./ELMo/"
CFORM_DIR     = "./targets/CForms.csv"
SYN_FIX_DIR   = "./targets/SynFix.csv"
LEX_FIX_DIR   = "./targets/LexFix.csv"
OVA_FIX_DIR   = "./targets/OvaFix.csv"
VECTORS_FILE  = "embeddings.csv"
W2V_RESULTS   = "W2V"
SCBOW_RESULTS = "SCBOW"
SKIP_RESULTS  = "SKIP"
ELMO_RESULTS  = "ELMO"
FILE_EXT      = ".model"

# Experiment Dirs
MODELS_DIR = "./SVM_Models/"
MODEL_EXT   = "_svm"

# Features:
USE_CFORM   = False
USE_SYN_FIX = False
USE_LEX_FIX = False
USE_OVA_FIX = False

# SVM Parameters: | Based on experiments by King and Cook (2018)
C_W2V   = 1
C_SKIP  = 1
C_SCBOW = 1
C_ELMo  = 1
KERNEL  = 'linear'
VERBOSE = True

# Shuffle Parameters:
RND_SEED = 42

def main():
    # Create Results Dir
    if not os.path.exists(os.path.dirname(MODELS_DIR)):
        try:
            os.makedirs(os.path.dirname(MODELS_DIR))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # -- EXTRACT DATASETS -- #
    # Extract all targets and remove those where classification is Q (unknown)
    targets = pd.read_csv(TARG_DIR, header=None, usecols=[0], sep=' ').values.flatten()
    indexes = np.where(targets != 'Q')
    targets_idiomatic = (targets[indexes] == 'I')
    targets_literal   = (targets[indexes] == 'L')

    features_w2v   = np.genfromtxt(W2V_DIR   + VECTORS_FILE, delimiter=',')[indexes]
    features_scbow = np.genfromtxt(SCBOW_DIR + VECTORS_FILE, delimiter=',')[indexes]
    features_skip  = np.genfromtxt(SKIP_DIR  + VECTORS_FILE, delimiter=',')[indexes]
    features_elmo  = np.genfromtxt(ELMO_DIR  + VECTORS_FILE, delimiter=',')[indexes]

    # -- ADD FAZLY's METRICS -- #
    if(USE_CFORM):
        cForms = np.genfromtxt(CFORM_DIR, delimiter=',')[indexes]
        cForms = cForms.reshape((cForms.size, 1))

        features_w2v   = np.append(features_w2v,   cForms, axis=1)
        features_scbow = np.append(features_scbow, cForms, axis=1)
        features_skip  = np.append(features_skip,  cForms, axis=1)
        features_elmo  = np.append(features_elmo,  cForms, axis=1)

    if(USE_SYN_FIX):
        synFix = np.genfromtxt(SYN_FIX_DIR, delimiter=',')[indexes]
        synFix = synFix.reshape((synFix.size, 1))

        features_w2v   = np.append(features_w2v,   synFix, axis=1)
        features_scbow = np.append(features_scbow, synFix, axis=1)
        features_skip  = np.append(features_skip,  synFix, axis=1)
        features_elmo  = np.append(features_elmo,  synFix, axis=1)

    if(USE_LEX_FIX):
        lexFix = np.genfromtxt(LEX_FIX_DIR, delimiter=',')[indexes]
        lexFix = lexFix.reshape((lexFix.size, 1))

        features_w2v   = np.append(features_w2v,   lexFix, axis=1)
        features_scbow = np.append(features_scbow, lexFix, axis=1)
        features_skip  = np.append(features_skip,  lexFix, axis=1)
        features_elmo  = np.append(features_elmo,  lexFix, axis=1)

    if(USE_OVA_FIX):
        ovaFix = np.genfromtxt(OVA_FIX_DIR, delimiter=',')[indexes]
        ovaFix = ovaFix.reshape((ovaFix.size, 1))

        features_w2v   = np.append(features_w2v,   ovaFix, axis=1)
        features_scbow = np.append(features_scbow, ovaFix, axis=1)
        features_skip  = np.append(features_skip,  ovaFix, axis=1)
        features_elmo  = np.append(features_elmo,  ovaFix, axis=1)

    # -- GENERATE PARTITIONS -- #
    w2v_X, scbow_X, skip_X, elmo_X, y = shuffle(features_w2v, features_scbow, features_skip, features_elmo, targets_idiomatic, random_state=RND_SEED)


    # -- WORD2VEC -- #
    print("<===================> Word2Vec <===================>")
    # Initialize Classifier
    svm_clf = svm.SVC(C=C_W2V, kernel=KERNEL, random_state=RND_SEED, verbose=VERBOSE)
    # Train Classifier
    svm_clf.fit(w2v_X, y)

    # Save Classifier
    modelName = os.path.join(MODELS_DIR, W2V_RESULTS + MODEL_EXT + FILE_EXT)
    with open(modelName, 'wb+') as modelFile:
        pickle.dump(svm_clf, modelFile, pickle.HIGHEST_PROTOCOL)


    # -- SIAMESE CBOW -- #
    print("<=================> Siamese CBOW <=================>")
    # Initialize Classifier
    svm_clf = svm.SVC(C=C_SCBOW, kernel=KERNEL, random_state=RND_SEED, verbose=VERBOSE)
    # Train Classifier
    svm_clf.fit(scbow_X, y)

    # Save Classifier
    modelName = os.path.join(MODELS_DIR, SCBOW_RESULTS + MODEL_EXT + FILE_EXT)
    with open(modelName, 'wb+') as modelFile:
        pickle.dump(svm_clf, modelFile, pickle.HIGHEST_PROTOCOL)


    # -- SKIP-THOUGHTS -- #
    print("<================> Skip - Thoughts <===============>")
    # Initialize Classifier
    svm_clf = svm.SVC(C=C_SKIP, kernel=KERNEL, random_state=RND_SEED, verbose=VERBOSE)
    # Train Classifier
    svm_clf.fit(skip_X, y)

    # Save Classifier
    modelName = os.path.join(MODELS_DIR, SKIP_RESULTS + MODEL_EXT + FILE_EXT)
    with open(modelName, 'wb+') as modelFile:
        pickle.dump(svm_clf, modelFile, pickle.HIGHEST_PROTOCOL)


    # -- ELMO -- #
    print("<=====================> ELMo <=====================>")
    # Initialize Classifier
    svm_clf = svm.SVC(C=C_ELMo, kernel=KERNEL, random_state=RND_SEED, verbose=VERBOSE)
    # Train Classifier
    svm_clf.fit(elmo_X, y)

    # Save Classifier
    modelName = os.path.join(MODELS_DIR, ELMO_RESULTS + MODEL_EXT + FILE_EXT)
    with open(modelName, 'wb+') as modelFile:
        pickle.dump(svm_clf, modelFile, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    if(args.TARG_DIR):
        TARG_DIR = args.TARG_DIR
    if(args.W2V_DIR):
        W2V_DIR = args.W2V_DIR
    if(args.SCBOW_DIR):
        SCBOW_DIR = args.SCBOW_DIR
    if(args.SKIP_DIR):
        SKIP_DIR = args.SKIP_DIR
    if(args.ELMO_DIR):
        ELMO_DIR = args.ELMO_DIR
    if(args.CFORM_DIR):
        CFORM_DIR = args.CFORM_DIR
    if(args.SYN_FIX_DIR):
        SYN_FIX_DIR = args.SYN_FIX_DIR
    if(args.LEX_FIX_DIR):
        LEX_FIX_DIR = args.LEX_FIX_DIR
    if(args.OVA_FIX_DIR):
        OVA_FIX_DIR = args.OVA_FIX_DIR
    if(args.VECTORS_FILE):
        VECTORS_FILE = args.VECTORS_FILE
    if(args.W2V_RESULTS):
        W2V_RESULTS = args.W2V_RESULTS
    if(args.SCBOW_RESULTS):
        SCBOW_RESULTS = args.SCBOW_RESULTS
    if(args.SKIP_RESULTS):
        SKIP_RESULTS = args.SKIP_RESULTS
    if(args.ELMO_RESULTS):
        ELMO_RESULTS = args.ELMO_RESULTS
    if(args.FILE_EXT):
        FILE_EXT = args.FILE_EXT

    if(args.MODELS_DIR):
        MODELS_DIR = args.MODELS_DIR
    if(args.MODEL_EXT):
        MODEL_EXT = args.MODEL_EXT

    if(args.USE_CFORM):
        USE_CFORM = True
    if(args.USE_SYN_FIX):
        USE_SYN_FIX = True
    if(args.USE_LEX_FIX):
        USE_LEX_FIX = True
    if(args.USE_OVA_FIX):
        USE_OVA_FIX = True

    if(args.C_W2V):
        C_W2V = args.C_W2V
    if(args.C_SKIP):
        C_SKIP = args.C_SKIP
    if(args.C_SCBOW):
        C_SCBOW = args.C_SCBOW
    if(args.C_ELMo):
        C_ELMo = args.C_ELMo
    if(args.KERNEL):
        KERNEL = args.KERNEL

    main()
