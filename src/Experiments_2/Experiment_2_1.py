"""
    File:   Experiment - Supervised [Fixedness Metrics + CForms]
    Author: Jose Juan Zavala Iglesias
    Date:   15/06/2019

    Experiment of a SVM using Fixedness Metrics and CForms as parameters.
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
CFORM_DIR     = "../targets/CForms.csv"
SYN_FIX_DIR   = "../targets/SynFix.csv"
LEX_FIX_DIR   = "../targets/LexFix.csv"
OVA_FIX_DIR   = "../targets/OvaFix.csv"
RESULTS_DIR   = "./results/"
FIX_RESULTS   = "FIXEDNESS"
IDIOMATIC_EXT = "_i"
LITERAL_EXT   = "_l"
FILE_EXT      = ".csv"

# SVM Parameters: | Based on experiments by King and Cook (2018)
PARAMS  = {'C':[0.01, 0.1, 1, 10, 100]}
KERNEL  = 'linear'
SCORING = ['accuracy', 'f1', 'precision', 'recall']

# K-Fold Cross Validation Parameters:
SPLITS    = 10
TEST_SIZE = 0.1
SEED      = 42
VERBOSE   = 4

# -- EXTRACT DATASETS -- #
# Extract all targets and remove those where classification is Q (unknown)
targets = pd.read_csv(TARGETS_DIR, header=None, usecols=[0], sep=' ').values.flatten()
indexes = np.where(targets != 'Q')
targets_idiomatic = (targets[indexes] == 'I')
targets_literal   = (targets[indexes] == 'L')

# -- CREATE FEATURE VECTORS -- #
cForms = np.genfromtxt(CFORM_DIR, delimiter=',')[indexes]
cForms = cForms.reshape((cForms.size, 1))

synFix = np.genfromtxt(SYN_FIX_DIR, delimiter=',')[indexes]
synFix = synFix.reshape((synFix.size, 1))

lexFix = np.genfromtxt(LEX_FIX_DIR, delimiter=',')[indexes]
lexFix = lexFix.reshape((lexFix.size, 1))

ovaFix = np.genfromtxt(OVA_FIX_DIR, delimiter=',')[indexes]
ovaFix = ovaFix.reshape((ovaFix.size, 1))

features_fix = np.append(cForms, synFix, axis=1)
features_fix = np.append(features_fix, lexFix, axis=1)
features_fix = np.append(features_fix, ovaFix, axis=1)

# -- GENERATE PARTITIONS -- #
cv = ShuffleSplit(n_splits=SPLITS, test_size=TEST_SIZE, random_state=SEED)

# -- INITIALIZE CLASSIFIER -- #
svm_clf = svm.SVC(kernel=KERNEL, random_state=SEED)

# -- FIXEDNESS CROSSVALIDATION -- #
print("<===================> FIXEDNESS <===================>")
# WITHOUT CFORM
# Idiomatic Detection
grid_fix    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
svm_fix     = grid_fix.fit(features_fix, targets_idiomatic)
results_fix = pd.DataFrame.from_dict(svm_fix.cv_results_)
results_fix.to_csv(RESULTS_DIR + FIX_RESULTS + IDIOMATIC_EXT + FILE_EXT)
# Literal Detection
grid_fix    = GridSearchCV(estimator=svm_clf, param_grid=PARAMS, scoring=SCORING, n_jobs=-1, cv=cv, return_train_score=True, verbose=VERBOSE, refit=False)
svm_fix     = grid_fix.fit(features_fix, targets_literal)
results_fix = pd.DataFrame.from_dict(svm_fix.cv_results_)
results_fix.to_csv(RESULTS_DIR + FIX_RESULTS + LITERAL_EXT + FILE_EXT)

