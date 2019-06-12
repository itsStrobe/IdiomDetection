"""
    File:   Experiment
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
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

# Files and Directories:
TARGETS_DIR   = "../targets/English_VNC_Cook/VNC-Tokens_cleaned"
W2V_DIR       = "../Word2Vec/"
SCBOW_DIR     = "../SiameseCBOW/"
SKIP_DIR      = "../SkipThoughts/"
ELMO_DIR      = "../ELMo/"
CFORM_DIR     = "../targets/CForms.csv"
VECTORS_FILE  = "embeddings.csv"
RESULTS_DIR   = "./results/"
W2V_RESULTS   = "W2V"
SCBOW_RESULTS = "SCBOW"
SKIP_RESULTS  = "SKIP"
ELMO_RESULTS  = "ELMO"
IDIOMATIC_EXT = "_i"
LITERAL_EXT   = "_l"
CFORM_EXT     = "_cform"
FILE_EXT      = ".csv"

# SVM Parameters: | Based on experiments by King and Cook (2018)
PARAMS  = {'C':[0.01, 0.1, 1, 10, 100]}
KERNEL  = 'linear'
SCORING = ['accuracy', 'f1', 'precision', 'recall']

# K-Fold Cross Validation Parameters:
SPLITS    = 10
TEST_SIZE = 0.1
SEED      = 5
VERBOSE   = 4

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

cforms = np.genfromtxt(CFORM_DIR, delimiter=',')[indexes]
cforms = cforms.reshape((cforms.size, 1))
features_w2v_cform   = np.append(features_w2v,   cforms, axis=1)
features_scbow_cform = np.append(features_scbow, cforms, axis=1) 
features_skip_cform  = np.append(features_skip,  cforms, axis=1)  
features_elmo_cform  = np.append(features_elmo,  cforms, axis=1)  

# -- GENERATE PARTITIONS -- #
cv = ShuffleSplit(n_splits=SPLITS, test_size=TEST_SIZE, random_state=SEED)

# -- INITIALIZE CLASSIFIER -- #
svm_clf = svm.SVC(kernel=KERNEL, random_state=SEED)

# -- WORD2VEC CROSSVALIDATION -- #
print("<===================> Word2Vec <===================>")
# WITHOUT CFORM
# Idiomatic Detection
grid_w2v    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
svm_w2v     = grid_w2v.fit(features_w2v, targets_idiomatic)
results_w2v = pd.DataFrame.from_dict(svm_w2v.cv_results_)
results_w2v.to_csv(RESULTS_DIR + W2V_RESULTS + IDIOMATIC_EXT + FILE_EXT)
# Literal Detection
grid_w2v    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
svm_w2v     = grid_w2v.fit(features_w2v, targets_literal)
results_w2v = pd.DataFrame.from_dict(svm_w2v.cv_results_)
results_w2v.to_csv(RESULTS_DIR + W2V_RESULTS + LITERAL_EXT + FILE_EXT)

# WITH CFORM
# Idiomatic Detection
grid_w2v    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
svm_w2v     = grid_w2v.fit(features_w2v_cform, targets_idiomatic)
results_w2v = pd.DataFrame.from_dict(svm_w2v.cv_results_)
results_w2v.to_csv(RESULTS_DIR + W2V_RESULTS + IDIOMATIC_EXT + CFORM_EXT + FILE_EXT)
# Literal Detection
grid_w2v    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
svm_w2v     = grid_w2v.fit(features_w2v_cform, targets_literal)
results_w2v = pd.DataFrame.from_dict(svm_w2v.cv_results_)
results_w2v.to_csv(RESULTS_DIR + W2V_RESULTS + LITERAL_EXT + CFORM_EXT + FILE_EXT)

# -- SIAMESE CBOW CROSSVALIDATION -- #
print("<=================> Siamese CBOW <=================>")
# WITHOUT CFORM
# Idiomatic Detection
grid_scbow    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
svm_scbow     = grid_scbow.fit(features_scbow, targets_idiomatic)
results_scbow = pd.DataFrame.from_dict(svm_scbow.cv_results_)
results_scbow.to_csv(RESULTS_DIR + SCBOW_RESULTS + IDIOMATIC_EXT + FILE_EXT)
# Literal Detection
grid_scbow    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
svm_scbow     = grid_scbow.fit(features_scbow, targets_literal)
results_scbow = pd.DataFrame.from_dict(svm_scbow.cv_results_)
results_scbow.to_csv(RESULTS_DIR + SCBOW_RESULTS + LITERAL_EXT + FILE_EXT)

# WITH CFORM
# Idiomatic Detection
grid_scbow    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
svm_scbow     = grid_scbow.fit(features_scbow_cform, targets_idiomatic)
results_scbow = pd.DataFrame.from_dict(svm_scbow.cv_results_)
results_scbow.to_csv(RESULTS_DIR + SCBOW_RESULTS + IDIOMATIC_EXT + CFORM_EXT + FILE_EXT)
# Literal Detection
grid_scbow    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
svm_scbow     = grid_scbow.fit(features_scbow_cform, targets_literal)
results_scbow = pd.DataFrame.from_dict(svm_scbow.cv_results_)
results_scbow.to_csv(RESULTS_DIR + SCBOW_RESULTS + LITERAL_EXT + CFORM_EXT + FILE_EXT)

# -- SKIP-THOUGHTS CROSSVALIDATION -- #
print("<================> Skip - Thoughts <===============>")
# WITHOUT CFORM
# Idiomatic Detection
grid_skip    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
svm_skip     = grid_skip.fit(features_skip, targets_idiomatic)
results_skip = pd.DataFrame.from_dict(svm_skip.cv_results_)
results_skip.to_csv(RESULTS_DIR + SKIP_RESULTS + IDIOMATIC_EXT + FILE_EXT)
# Literal Detection
grid_skip    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
svm_skip     = grid_skip.fit(features_skip, targets_literal)
results_skip = pd.DataFrame.from_dict(svm_skip.cv_results_)
results_skip.to_csv(RESULTS_DIR + SKIP_RESULTS + LITERAL_EXT + FILE_EXT)

# WITH CFORM
# Idiomatic Detection
grid_skip    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
svm_skip     = grid_skip.fit(features_skip_cform, targets_idiomatic)
results_skip = pd.DataFrame.from_dict(svm_skip.cv_results_)
results_skip.to_csv(RESULTS_DIR + SKIP_RESULTS + IDIOMATIC_EXT + CFORM_EXT + FILE_EXT)
# Literal Detection
grid_skip    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
svm_skip     = grid_skip.fit(features_skip_cform, targets_literal)
results_skip = pd.DataFrame.from_dict(svm_skip.cv_results_)
results_skip.to_csv(RESULTS_DIR + SKIP_RESULTS + LITERAL_EXT + CFORM_EXT + FILE_EXT)

# -- ELMO CROSSVALIDATION -- #
print("<=====================> ELMo <=====================>")
# WITHOUT CFORM
# Idiomatic Detection
grid_elmo    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
svm_elmo     = grid_elmo.fit(features_elmo, targets_idiomatic)
results_skip = pd.DataFrame.from_dict(svm_elmo.cv_results_)
results_skip.to_csv(RESULTS_DIR + ELMO_RESULTS + IDIOMATIC_EXT + FILE_EXT)
# Literal Detection
grid_elmo    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
svm_elmo     = grid_elmo.fit(features_elmo, targets_literal)
results_skip = pd.DataFrame.from_dict(svm_elmo.cv_results_)
results_skip.to_csv(RESULTS_DIR + ELMO_RESULTS + LITERAL_EXT + FILE_EXT)

# WITH CFORM
# Idiomatic Detection
grid_elmo    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
svm_elmo     = grid_elmo.fit(features_elmo_cform, targets_idiomatic)
results_skip = pd.DataFrame.from_dict(svm_elmo.cv_results_)
results_skip.to_csv(RESULTS_DIR + ELMO_RESULTS + IDIOMATIC_EXT + CFORM_EXT + FILE_EXT)
# Literal Detection
grid_elmo    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
svm_elmo     = grid_elmo.fit(features_elmo_cform, targets_literal)
results_skip = pd.DataFrame.from_dict(svm_elmo.cv_results_)
results_skip.to_csv(RESULTS_DIR + ELMO_RESULTS + LITERAL_EXT + CFORM_EXT + FILE_EXT)
