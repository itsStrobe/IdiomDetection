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
from sklearn.utils import shuffle
from sklearn.model_selection import ShuffleSplit, GridSearchCV, train_test_split
from sklearn.decomposition import PCA

# Files and Directories:
OG_SENT_DIR   = "../targets/Extracted_Sentences.txt"
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

# Experiment Suffix
EXP_EXT       = "_fixedness_cform"

# File Extensions
FILE_EXT      = ".csv"
IMG_EXT       = ".png"

# Features:
USE_CFORM   = True
USE_SYN_FIX = True
USE_LEX_FIX = True
USE_OVA_FIX = True

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

def gen_plot(feat, targ, pred, title_targ, title_pred, saveDir, dispPlot=False):
    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(feat)

    featDf = pd.DataFrame(data = principalComponents, columns = ['X', 'Y'])
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
        ax_targ.scatter(targ_finalDf.loc[indicesToKeep, 'X'],
                        targ_finalDf.loc[indicesToKeep, 'Y'],
                        c = color,
                        s = 10)

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
        ax_pred.scatter(pred_finalDf.loc[indicesToKeep, 'X'],
                        pred_finalDf.loc[indicesToKeep, 'Y'],
                        c = color,
                        s = 10)

    ax_pred.legend(targets)
    ax_pred.grid()

    plt.savefig(saveDir)

    if(dispPlot):
        plt.show()

def saveClassifiedSentences(all_sent, all_targ, all_pred, fileDir):
#    for sent, targ, pred in zip(all_sent, all_targ, all_pred):
#        if(targ != pred):
#            print("Sentence", sent, "| Target:", targ, "| Clustered As:", pred)

    all_sent = all_sent.reshape((all_sent.size, 1))
    all_targ = all_targ.reshape((all_targ.size, 1))
    all_pred = all_pred.reshape((all_pred.size, 1))
    data = np.append(all_sent, all_targ, axis=1)
    data = np.append(data, all_pred, axis = 1)
    pd.DataFrame(data = data, columns=['Sentence', 'Target', 'Prediction']).to_csv(fileDir, sep='\t')

# -- EXTRACT DATASETS -- #
# Extract all targets and remove those where classification is Q (unknown)
targets = pd.read_csv(TARGETS_DIR, header=None, usecols=[0], sep=' ').values.flatten()
indexes = np.where(targets != 'Q')

targets_idiomatic = (targets[indexes] == 'I')
targets_literal   = (targets[indexes] == 'L')

# Original Sentences
og_sent = np.genfromtxt(OG_SENT_DIR, delimiter="\t", dtype=None, encoding="utf_8")[indexes]

# Sentence Embeddings
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

# Split Sets:
sent_X, w2v_X, scbow_X, skip_X, elmo_X, y = shuffle(og_sent, features_w2v_fix, features_scbow_fix, features_skip_fix, features_elmo_fix, targets_idiomatic, random_state=RND_STATE)

# -- Extract Random Centroids -- #
# Initialize Cluster Vectors
w2v_centroids   = np.zeros((N_CLUSTERS, w2v_X.shape[1]))
scbow_centroids = np.zeros((N_CLUSTERS, scbow_X.shape[1]))
skip_centroids  = np.zeros((N_CLUSTERS, skip_X.shape[1]))
elmo_centroids  = np.zeros((N_CLUSTERS, elmo_X.shape[1]))

# Extract Centroids Indexes
np.random.seed(RND_SEED)
cent_i = np.random.choice(np.where(y == True)[0])
cent_l = np.random.choice(np.where(y == False)[0])

# Get Centroids
w2v_centroids[0]   = w2v_X[cent_i]
w2v_centroids[1]   = w2v_X[cent_l]

scbow_centroids[0] = scbow_X[cent_i]
scbow_centroids[1] = scbow_X[cent_l]

skip_centroids[0]  = skip_X[cent_i]
skip_centroids[1]  = skip_X[cent_l]

elmo_centroids[0]  = elmo_X[cent_i]
elmo_centroids[1]  = elmo_X[cent_l]

# -- Train k-Means -- #

print("<===================> Word2Vec <===================>")
# - Run KMeans
w2v_kMeans = KMeans(n_clusters=N_CLUSTERS, init=w2v_centroids, n_init=N_INIT, n_jobs=N_JOBS, max_iter=MAX_ITER, verbose=VERBOSE, random_state=RND_STATE).fit(w2v_X)

# Display Clusters:
w2v_clust_labels = (np.array(w2v_kMeans.labels_) == y[cent_i])
gen_plot(w2v_X, y, w2v_clust_labels, "Original Word2Vec Labels", "k-Means Labels", RESULTS_DIR + W2V_RESULTS + EXP_EXT + IMG_EXT)
saveClassifiedSentences(sent_X, y, w2v_clust_labels, RESULTS_DIR + W2V_RESULTS + EXP_EXT + FILE_EXT)

print("Results:", classification_report(y, w2v_clust_labels))

print("<=================> Siamese CBOW <=================>")
# - Run KMeans
scbow_kMeans = KMeans(n_clusters=N_CLUSTERS, init=scbow_centroids, n_init=N_INIT, n_jobs=N_JOBS, max_iter=MAX_ITER, verbose=VERBOSE, random_state=RND_STATE).fit(w2v_X)

# Display Clusters:
scbow_clust_labels = (np.array(scbow_kMeans.labels_) == y[cent_i])
gen_plot(scbow_X, y, scbow_clust_labels, "Original Siamese CBOW Labels", "k-Means Labels", RESULTS_DIR + SCBOW_RESULTS + EXP_EXT + IMG_EXT)
saveClassifiedSentences(sent_X, y, scbow_clust_labels, RESULTS_DIR + SCBOW_RESULTS + EXP_EXT + FILE_EXT)

print("Results:", classification_report(y, scbow_clust_labels))

print("<================> Skip - Thoughts <===============>")
# - Run KMeans
skip_kMeans = KMeans(n_clusters=N_CLUSTERS, init=skip_centroids, n_init=N_INIT, n_jobs=N_JOBS, max_iter=MAX_ITER, verbose=VERBOSE, random_state=RND_STATE).fit(skip_X)

# Display Clusters:
skip_clust_labels = (np.array(skip_kMeans.labels_) == y[cent_i])
gen_plot(skip_X, y, skip_clust_labels, "Original Skip-Thoughts Labels", "k-Means Labels", RESULTS_DIR + SKIP_RESULTS + EXP_EXT + IMG_EXT)
saveClassifiedSentences(sent_X, y, skip_clust_labels, RESULTS_DIR + SKIP_RESULTS + EXP_EXT + FILE_EXT)

print("Results:", classification_report(y, skip_clust_labels))

print("<=====================> ELMo <=====================>")
# - Run KMeans
elmo_kMeans = KMeans(n_clusters=N_CLUSTERS, init=elmo_centroids, n_init=N_INIT, n_jobs=N_JOBS, max_iter=MAX_ITER, verbose=VERBOSE, random_state=RND_STATE).fit(elmo_X)

# Display Clusters:
elmo_clust_labels = (np.array(elmo_kMeans.labels_) == y[cent_i])
gen_plot(elmo_X, y, elmo_clust_labels, "Original ELMo Labels", "k-Means Labels", RESULTS_DIR + ELMO_RESULTS + EXP_EXT + IMG_EXT)
saveClassifiedSentences(sent_X, y, elmo_clust_labels, RESULTS_DIR + ELMO_RESULTS + EXP_EXT + FILE_EXT)

print("Results:", classification_report(y, elmo_clust_labels))
