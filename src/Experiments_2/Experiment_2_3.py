"""
    File:   Experiment
    Author: Jose Juan Zavala Iglesias
    Date:   16/06/2019

    k-Means Classification evaluation for VNC Tokens Dataset using word embeddings generated with:
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import ShuffleSplit, GridSearchCV, train_test_split
from sklearn.decomposition import PCA

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
RESULTS_DIR   = "./results/"
W2V_RESULTS   = "W2V"
SCBOW_RESULTS = "SCBOW"
SKIP_RESULTS  = "SKIP"
ELMO_RESULTS  = "ELMO"
IDIOMATIC_EXT = "_i"
LITERAL_EXT   = "_l"
EXP_EXT       = "_fixedness_cform"
FILE_EXT      = ".csv"
IMG_EXT       = ".png"

# k-Means Parameters:
N_CLUSTERS = 2
N_INIT     = 10
MAX_ITER   = 300
N_JOBS     = -1
VERBOSE    = 1

# Split Parameters
RND_STATE = 42
TEST_SIZE = 0.3
SHUFFLE   = True

# Cluster Initialization
RND_SEED = 42

def gen_plot(feat, targ, pred, title_targ, title_pred, saveDir):
    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(feat)

    featDf = pd.DataFrame(data = principalComponents
                 , columns = ['X', 'Y'])
    targDf = pd.DataFrame(data = targ, columns=['target'])
    predDf = pd.DataFrame(data = pred, columns=['target'])

    targ_finalDf = pd.concat([featDf, targDf[['target']]], axis = 1)
    pred_finalDf = pd.concat([featDf, predDf[['target']]], axis = 1)

    fig = plt.figure(figsize = (10,10))

    ax_targ = fig.add_subplot(2,1,1) 
    ax_targ.set_xlabel('X', fontsize = 15)
    ax_targ.set_ylabel('Y', fontsize = 15)
    ax_targ.set_title(title_targ, fontsize = 20)

    targets = [True, False]
    colors = ['b', 'r']
    for target, color in zip(targets,colors):
        indicesToKeep = targ_finalDf['target'] == target
        ax_targ.scatter(targ_finalDf.loc[indicesToKeep, 'X']
                   , targ_finalDf.loc[indicesToKeep, 'Y']
                   , c = color
                   , s = 50)

    ax_targ.legend(targets)
    ax_targ.grid()

    ax_pred = fig.add_subplot(2,1,2) 
    ax_pred.set_xlabel('X', fontsize = 15)
    ax_pred.set_ylabel('Y', fontsize = 15)
    ax_pred.set_title(title_pred, fontsize = 20)

    targets = [True, False]
    colors = ['b', 'r']
    for target, color in zip(targets,colors):
        indicesToKeep = pred_finalDf['target'] == target
        ax_pred.scatter(pred_finalDf.loc[indicesToKeep, 'X']
                   , targ_finalDf.loc[indicesToKeep, 'Y']
                   , c = color
                   , s = 50)

    ax_pred.legend(targets)
    ax_pred.grid()

    plt.savefig(saveDir)
    plt.show()

# -- EXTRACT DATASETS -- #
# Extract all targets and remove those where classification is Q (unknown)
targets = pd.read_csv(TARGETS_DIR, header=None, usecols=[0], sep=' ').values.flatten()
indexes = np.where(targets != 'Q')
targets_test = targets[indexes]
targets_idiomatic = (targets[indexes] == 'I')
targets_literal   = (targets[indexes] == 'L')

features_w2v   = np.genfromtxt(W2V_DIR   + VECTORS_FILE, delimiter=',')[indexes]
features_scbow = np.genfromtxt(SCBOW_DIR + VECTORS_FILE, delimiter=',')[indexes]
features_skip  = np.genfromtxt(SKIP_DIR  + VECTORS_FILE, delimiter=',')[indexes]
features_elmo  = np.genfromtxt(ELMO_DIR  + VECTORS_FILE, delimiter=',')[indexes]

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

features_w2v_fix   = np.append(features_w2v,   features_fix, axis=1)
features_scbow_fix = np.append(features_scbow, features_fix, axis=1) 
features_skip_fix  = np.append(features_skip,  features_fix, axis=1)  
features_elmo_fix  = np.append(features_elmo,  features_fix, axis=1)

# Split Sets:
print(features_w2v_fix.shape, targets_idiomatic.shape)
w2v_X_train, w2v_X_test, w2v_y_train, w2v_y_test         = train_test_split(features_w2v_fix,   targets_idiomatic, test_size=TEST_SIZE, shuffle=SHUFFLE, random_state=RND_STATE)
scbow_X_train, scbow_X_test, scbow_y_train, scbow_y_test = train_test_split(features_scbow_fix, targets_idiomatic, test_size=TEST_SIZE, shuffle=SHUFFLE, random_state=RND_STATE)
skip_X_train, skip_X_test, skip_y_train, skip_y_test     = train_test_split(features_skip_fix,  targets_idiomatic, test_size=TEST_SIZE, shuffle=SHUFFLE, random_state=RND_STATE)
elmo_X_train, elmo_X_test, elmo_y_train, elmo_y_test     = train_test_split(features_elmo_fix,  targets_idiomatic, test_size=TEST_SIZE, shuffle=SHUFFLE, random_state=RND_STATE)

# -- Extract Random Centroids -- #
# Initialize Cluster Vectors
w2v_centroids   = np.zeros((N_CLUSTERS, w2v_X_train.shape[1]))
scbow_centroids = np.zeros((N_CLUSTERS, scbow_X_train.shape[1]))
skip_centroids  = np.zeros((N_CLUSTERS, skip_X_train.shape[1]))
elmo_centroids  = np.zeros((N_CLUSTERS, elmo_X_train.shape[1]))

# Extract Centroids Indexes
np.random.seed(RND_SEED)
cent_i = np.random.choice(np.where(w2v_y_train == True)[0])
cent_l = np.random.choice(np.where(w2v_y_train == False)[0])

# Get Centroids
w2v_centroids[0]   = w2v_X_train[cent_i]
w2v_centroids[1]   = w2v_X_train[cent_l]

scbow_centroids[0] = scbow_X_train[cent_i]
scbow_centroids[1] = scbow_X_train[cent_l]

skip_centroids[0]  = skip_X_train[cent_i]
skip_centroids[1]  = skip_X_train[cent_l]

elmo_centroids[0]  = elmo_X_train[cent_i]
elmo_centroids[1]  = elmo_X_train[cent_l]

# -- Train k-Means -- #

print("<===================> Word2Vec <===================>")
# - Run KMeans
w2v_kMeans = KMeans(n_clusters=N_CLUSTERS, init=w2v_centroids, n_init=N_INIT, n_jobs=N_JOBS, max_iter=MAX_ITER, verbose=VERBOSE, random_state=RND_STATE).fit(w2v_X_train)

# - Predictions
# Cluster Indexes:
w2v_idx = w2v_kMeans.predict(w2v_centroids)

if(w2v_idx[0] == w2v_idx[1]):
    print("Both centroid initializers are in same Cluster")

print("Idiomatic Centroid:", w2v_idx[0])
print("Literal Centroid:", w2v_idx[1])
w2v_pred_test = (w2v_kMeans.predict(w2v_X_test) == w2v_idx[0])

correct = 0
total   = 0
pred_i  = 0
pred_l  = 0
for pred, real in zip(w2v_pred_test, w2v_y_test):
    if(pred):
        pred_i += 1
    else:
        pred_l += 1

    if(pred == real):
        correct += 1
    total += 1

print("Results:", classification_report(w2v_y_test, w2v_pred_test))
print("Pred_i:", pred_i)
print("Pred_l:", pred_l)

gen_plot(w2v_X_test, w2v_y_test, w2v_pred_test, "W2V Targets", "W2V Predictions", RESULTS_DIR + W2V_RESULTS + EXP_EXT + IMG_EXT)

print("<=================> Siamese CBOW <=================>")
# - Run KMeans
scbow_kMeans = KMeans(n_clusters=N_CLUSTERS, init=scbow_centroids, n_init=N_INIT, n_jobs=N_JOBS, max_iter=MAX_ITER, verbose=VERBOSE, random_state=RND_STATE).fit(scbow_X_train)

# - Predictions
# Cluster Indexes:
scbow_idx = scbow_kMeans.predict(scbow_centroids)

if(scbow_idx[0] == scbow_idx[1]):
    print("Both centroid initializers are in same Cluster")

print("Idiomatic Centroid:", scbow_idx[0])
print("Literal Centroid:", scbow_idx[1])
scbow_pred_test = (scbow_kMeans.predict(scbow_X_test) == scbow_idx[0])

correct = 0
total   = 0
pred_i  = 0
pred_l  = 0
for pred, real in zip(scbow_pred_test, scbow_y_test):
    if(pred):
        pred_i += 1
    else:
        pred_l += 1

    if(pred == real):
        correct += 1
    total += 1

print("Results:", classification_report(scbow_y_test, scbow_pred_test))
print("Pred_i:", pred_i)
print("Pred_l:", pred_l)

gen_plot(scbow_X_test, scbow_y_test, scbow_pred_test, "Siamese CBOW Targets", "Siamese CBOW Predictions", RESULTS_DIR + SCBOW_RESULTS + EXP_EXT + IMG_EXT)

print("<================> Skip - Thoughts <===============>")
# - Run KMeans
skip_kMeans = KMeans(n_clusters=N_CLUSTERS, init=skip_centroids, n_init=N_INIT, n_jobs=N_JOBS, max_iter=MAX_ITER, verbose=VERBOSE, random_state=RND_STATE).fit(skip_X_train)

# - Predictions
# Cluster Indexes:
skip_idx = skip_kMeans.predict(skip_centroids)

if(skip_idx[0] == skip_idx[1]):
    print("Both centroid initializers are in same Cluster")

print("Idiomatic Centroid:", skip_idx[0])
print("Literal Centroid:", skip_idx[1])
skip_pred_test = (skip_kMeans.predict(skip_X_test) == skip_idx[0])

correct = 0
total   = 0
pred_i  = 0
pred_l  = 0
for pred, real in zip(skip_pred_test, skip_y_test):
    if(pred):
        pred_i += 1
    else:
        pred_l += 1

    if(pred == real):
        correct += 1
    total += 1

print("Results:", classification_report(skip_y_test, skip_pred_test))
print("Pred_i:", pred_i)
print("Pred_l:", pred_l)

gen_plot(skip_X_test, skip_y_test, skip_pred_test, "Skip-Thoughts Targets", "Skip-Thoughts Predictions", RESULTS_DIR + SKIP_RESULTS + EXP_EXT + IMG_EXT)

print("<=====================> ELMo <=====================>")
# - Run KMeans
elmo_kMeans = KMeans(n_clusters=N_CLUSTERS, init=elmo_centroids, n_init=N_INIT, n_jobs=N_JOBS, max_iter=MAX_ITER, verbose=VERBOSE, random_state=RND_STATE).fit(elmo_X_train)

# - Predictions
# Cluster Indexes:
elmo_idx = elmo_kMeans.predict(elmo_centroids)

if(elmo_idx[0] == elmo_idx[1]):
    print("Both centroid initializers are in same Cluster")

print("Idiomatic Centroid:", elmo_idx[0])
print("Literal Centroid:", elmo_idx[1])
elmo_pred_test = (elmo_kMeans.predict(elmo_X_test) == elmo_idx[0])

correct = 0
total   = 0
pred_i  = 0
pred_l  = 0
for pred, real in zip(elmo_pred_test, elmo_y_test):
    if(pred):
        pred_i += 1
    else:
        pred_l += 1

    if(pred == real):
        correct += 1
    total += 1

print("Results:", classification_report(elmo_y_test, elmo_pred_test))
print("Pred_i:", pred_i)
print("Pred_l:", pred_l)

gen_plot(elmo_X_test, elmo_y_test, elmo_pred_test, "ELMo Targets", "ELMo Predictions", RESULTS_DIR + ELMO_RESULTS + EXP_EXT + IMG_EXT)
