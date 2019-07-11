"""
    File:   Experiment_2_2
    Author: Jose Juan Zavala Iglesias
    Date:   03/06/2019

    SVM Classification evaluation for VNC Tokens Dataset using word embeddings generated with:
        >Word2Vec
        >Skip-Thoughts
        >Siamese CBOW
        >ELMo
"""

import os
import re
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 
from sklearn.model_selection import ShuffleSplit, GridSearchCV

# ------------- ARGS ------------- #
parser = argparse.ArgumentParser()

parser.add_argument("--TARGETS_DIR"   , "--targets_directory"                 , type=str, help="Location of the File Containing the Targets in VNC-Token format.")
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
parser.add_argument("--IDIOMATIC_EXT" , "--idiomatic_results_suffix"          , type=str, help="Filename Extension for Idiomatic Test Results.")
parser.add_argument("--LITERAL_EXT"   , "--literal_results_suffix"            , type=str, help="Filename Extension for Literal Test Results.")
parser.add_argument("--FILE_EXT"      , "--output_file_extension"             , type=str, help="File Extension for Output Files.")

parser.add_argument("--RESULTS_DIR" , "--results_directory" , type=str, help="Results Directory.")
parser.add_argument("--EXP_EXT"     , "--experiment_suffix" , type=str, help="Experiments Name Extension.")

parser.add_argument("--USE_CFORM"   , help="Use flag to indicate if CForm Feature Should Be Added.", action="store_true")
parser.add_argument("--USE_SYN_FIX" , help="Use flag to indicate if Syntactic Fixedness Feature Should Be Added.", action="store_true")
parser.add_argument("--USE_LEX_FIX" , help="Use flag to indicate if Lexical Fixedness Feature Should Be Added.", action="store_true")
parser.add_argument("--USE_OVA_FIX" , help="Use flag to indicate if Overall Fixedness Feature Should Be Added.", action="store_true")

args = parser.parse_args()
# ------------- ARGS ------------- #

# Files and Directories:
TARGETS_DIR   = "../targets/English_VNC_Cook/VNC-Tokens_cleaned"
W2V_DIR       = "../Word2Vec/"
SCBOW_DIR     = "../SiameseCBOW/"
SKIP_DIR      = "../SkipThoughts/"
ELMO_DIR      = "../ELMo/"
CFORM_DIR     = "../targets/CForms.csv"
SYN_FIX_DIR   = "../targets/SynFix.csv"
LEX_FIX_DIR   = "../targets/LexFix.csv"
OVA_FIX_DIR   = "../targets/OvaFix.csv"
VECTORS_FILE  = "embeddings.csv"
W2V_RESULTS   = "W2V"
SCBOW_RESULTS = "SCBOW"
SKIP_RESULTS  = "SKIP"
ELMO_RESULTS  = "ELMO"
IDIOMATIC_EXT = "_i"
LITERAL_EXT   = "_l"
FILE_EXT      = ".csv"

# Experiment Dirs
RESULTS_DIR = "./results/Experiment_2_2/"
EXP_EXT     = ""

# Features:
USE_CFORM   = False
USE_SYN_FIX = False
USE_LEX_FIX = False
USE_OVA_FIX = False

# SVM Parameters: | Based on experiments by King and Cook (2018)
PARAMS  = {'C':[0.01, 0.1, 1, 10, 100]}
KERNEL  = 'linear'
SCORING = ['accuracy', 'f1', 'precision', 'recall']

# K-Fold Cross Validation Parameters:
SPLITS    = 10
TEST_SIZE = 0.1
SEED      = 42
VERBOSE   = 4

def main():
    # Create Results Dir
    if not os.path.exists(os.path.dirname(RESULTS_DIR)):
        try:
            os.makedirs(os.path.dirname(RESULTS_DIR))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # -- EXTRACT DATASETS -- #
    # Extract all targets and remove those where classification is Q (unknown)
    targets = pd.read_csv(TARGETS_DIR, header=None, usecols=[0], sep=' ').values.flatten()
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
    cv = ShuffleSplit(n_splits=SPLITS, test_size=TEST_SIZE, random_state=SEED)

    # -- INITIALIZE CLASSIFIER -- #
    svm_clf = svm.SVC(kernel=KERNEL, random_state=SEED)

    # -- WORD2VEC CROSSVALIDATION -- #
    print("<===================> Word2Vec <===================>")
    # Idiomatic Detection
    grid_w2v    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
    svm_w2v     = grid_w2v.fit(features_w2v, targets_idiomatic)
    results_w2v = pd.DataFrame.from_dict(svm_w2v.cv_results_)
    results_w2v.to_csv(RESULTS_DIR + W2V_RESULTS + IDIOMATIC_EXT + EXP_EXT + FILE_EXT)
    # Literal Detection
    grid_w2v    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
    svm_w2v     = grid_w2v.fit(features_w2v, targets_literal)
    results_w2v = pd.DataFrame.from_dict(svm_w2v.cv_results_)
    results_w2v.to_csv(RESULTS_DIR + W2V_RESULTS + LITERAL_EXT + EXP_EXT + FILE_EXT)

    # -- SIAMESE CBOW CROSSVALIDATION -- #
    print("<=================> Siamese CBOW <=================>")
    # Idiomatic Detection
    grid_scbow    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
    svm_scbow     = grid_scbow.fit(features_scbow, targets_idiomatic)
    results_scbow = pd.DataFrame.from_dict(svm_scbow.cv_results_)
    results_scbow.to_csv(RESULTS_DIR + SCBOW_RESULTS + IDIOMATIC_EXT + EXP_EXT + FILE_EXT)
    # Literal Detection
    grid_scbow    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
    svm_scbow     = grid_scbow.fit(features_scbow, targets_literal)
    results_scbow = pd.DataFrame.from_dict(svm_scbow.cv_results_)
    results_scbow.to_csv(RESULTS_DIR + SCBOW_RESULTS + LITERAL_EXT + EXP_EXT + FILE_EXT)

    # -- SKIP-THOUGHTS CROSSVALIDATION -- #
    print("<================> Skip - Thoughts <===============>")
    # Idiomatic Detection
    grid_skip    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
    svm_skip     = grid_skip.fit(features_skip, targets_idiomatic)
    results_skip = pd.DataFrame.from_dict(svm_skip.cv_results_)
    results_skip.to_csv(RESULTS_DIR + SKIP_RESULTS + IDIOMATIC_EXT + EXP_EXT + FILE_EXT)
    # Literal Detection
    grid_skip    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
    svm_skip     = grid_skip.fit(features_skip, targets_literal)
    results_skip = pd.DataFrame.from_dict(svm_skip.cv_results_)
    results_skip.to_csv(RESULTS_DIR + SKIP_RESULTS + LITERAL_EXT + EXP_EXT + FILE_EXT)

    # -- ELMO CROSSVALIDATION -- #
    print("<=====================> ELMo <=====================>")
    # Idiomatic Detection
    grid_elmo    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
    svm_elmo     = grid_elmo.fit(features_elmo, targets_idiomatic)
    results_skip = pd.DataFrame.from_dict(svm_elmo.cv_results_)
    results_skip.to_csv(RESULTS_DIR + ELMO_RESULTS + IDIOMATIC_EXT + EXP_EXT + FILE_EXT)
    # Literal Detection
    grid_elmo    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
    svm_elmo     = grid_elmo.fit(features_elmo, targets_literal)
    results_skip = pd.DataFrame.from_dict(svm_elmo.cv_results_)
    results_skip.to_csv(RESULTS_DIR + ELMO_RESULTS + LITERAL_EXT + EXP_EXT + FILE_EXT)

if __name__ == '__main__':

    if(args.TARGETS_DIR):
        TARGETS_DIR = args.TARGETS_DIR
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
    if(args.IDIOMATIC_EXT):
        IDIOMATIC_EXT = args.IDIOMATIC_EXT
    if(args.LITERAL_EXT):
        LITERAL_EXT = args.LITERAL_EXT
    if(args.FILE_EXT):
        FILE_EXT = args.FILE_EXT

    if(args.RESULTS_DIR):
        RESULTS_DIR = args.RESULTS_DIR
    if(args.EXP_EXT):
        EXP_EXT = args.EXP_EXT

    if(args.USE_CFORM):
        USE_CFORM = True
    if(args.USE_SYN_FIX):
        USE_SYN_FIX = True
    if(args.USE_LEX_FIX):
        USE_LEX_FIX = True
    if(args.USE_OVA_FIX):
        USE_OVA_FIX = True

    main()
