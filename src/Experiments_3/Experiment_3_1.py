"""
    File:   Experiment
    Author: Jose Juan Zavala Iglesias
    Date:   23/06/2019

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

import UnsupervisedMetrics

# Files and Directories:
OG_SENT_DIR      = "../targets/Extracted_Sentences_cand.txt"
TARGETS_DIR      = "../targets/VNC-Tokens_candidates"
W2V_DIR          = "../Word2Vec/"
SCBOW_DIR        = "../SiameseCBOW/"
SKIP_DIR         = "../SkipThoughts/"
ELMO_DIR         = "../ELMo/"
OVA_FIX_DIR      = "../targets/OvaFix_cand.csv"
VECTORS_FILE     = "embeddings_cand.csv"
VECTORS_FILE_VNC = "embeddings_VNC_cand.csv"
W2V_RESULTS      = "W2V"
SCBOW_RESULTS    = "SCBOW"
SKIP_RESULTS     = "SKIP"
ELMO_RESULTS     = "ELMO"

# Experiment Dirs
RESULTS_DIR = "./results/Experiment_3_1/"
EXP_EXT     = "_cosineSimilarity"

# File Extensions
FILE_EXT = ".tsv"
CSV_EXT  = ".csv"
IMG_EXT  = ".png"

# Unsupervised Parameters:
COS_DIST_T  = 0.6
COS_DIST_Op = '<'

# Shuffle Parameters
RND_STATE = 42

# Other Parameters
SAVE_PLT = False

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

    plt.close()

def saveClassifiedSentences(all_sent, all_cSim, all_pred, fileDir):

    all_sent = all_sent.reshape((all_sent.size, 1))
    all_cSim = all_cSim.reshape((all_cSim.size, 1))
    all_pred = all_pred.reshape((all_pred.size, 1))
    data = np.append(all_sent, all_cSim, axis=1)
    data = np.append(data, all_pred, axis = 1)
    pd.DataFrame(data = data, columns=['Sentence', 'Unsupervised Metric', 'Classification']).to_csv(fileDir, sep='\t')

# Create Results Dir
if not os.path.exists(os.path.dirname(RESULTS_DIR)):
    try:
        os.makedirs(os.path.dirname(RESULTS_DIR))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

# -- EXTRACT DATASETS -- #
targets = pd.read_csv(TARGETS_DIR, header=None, usecols=[0], sep=' ').values.flatten()

# Original Sentences
og_sent = np.genfromtxt(OG_SENT_DIR, delimiter="\t", dtype=None, encoding="utf_8")

# Sentence Embeddings
features_w2v   = np.genfromtxt(W2V_DIR   + VECTORS_FILE, delimiter=',')
features_scbow = np.genfromtxt(SCBOW_DIR + VECTORS_FILE, delimiter=',')
features_skip  = np.genfromtxt(SKIP_DIR  + VECTORS_FILE, delimiter=',')
features_elmo  = np.genfromtxt(ELMO_DIR  + VECTORS_FILE, delimiter=',')

# VNC Embeddings
features_w2v_VNC   = np.genfromtxt(W2V_DIR   + VECTORS_FILE_VNC, delimiter=',')
features_scbow_VNC = np.genfromtxt(SCBOW_DIR + VECTORS_FILE_VNC, delimiter=',')
features_skip_VNC  = np.genfromtxt(SKIP_DIR  + VECTORS_FILE_VNC, delimiter=',')
features_elmo_VNC  = np.genfromtxt(ELMO_DIR  + VECTORS_FILE_VNC, delimiter=',')

# Shuffle Sets:
sent_X, w2v_X, w2v_X_VNC, scbow_X, scbow_X_VNC, skip_X, skip_X_VNC, elmo_X, elmo_X_VNC = shuffle(og_sent, features_w2v, features_w2v_VNC, features_scbow, features_scbow_VNC, features_skip, features_skip_VNC, features_elmo, features_elmo_VNC, random_state=RND_STATE)


print("<===================> Word2Vec <===================>")
# - Calculate Cosine Similarity
w2v_cosSims = UnsupervisedMetrics.CosineSimilarity(w2v_X, w2v_X_VNC)

# - Get Predictions
w2v_pred = UnsupervisedMetrics.ThresholdClassifier(w2v_cosSims, T=COS_DIST_T, Op=COS_DIST_Op)

# Display Classifications:
if(SAVE_PLT): gen_plot(w2v_X, y, w2v_pred, "Original Word2Vec Labels", "Cosine Similarity Labels", RESULTS_DIR + W2V_RESULTS + EXP_EXT + IMG_EXT)
saveClassifiedSentences(sent_X, w2v_cosSims, w2v_pred, RESULTS_DIR + W2V_RESULTS + EXP_EXT + FILE_EXT)

print("<=================> Siamese CBOW <=================>")
# - Calculate Cosine Similarity
scbow_cosSims = UnsupervisedMetrics.CosineSimilarity(scbow_X, scbow_X_VNC)

# - Get Predictions
scbow_pred = UnsupervisedMetrics.ThresholdClassifier(scbow_cosSims, T=COS_DIST_T, Op=COS_DIST_Op)

# Display Classifications:
if(SAVE_PLT): gen_plot(scbow_X, y, scbow_pred, "Original Siamese CBOW Labels", "Cosine Similarity Labels", RESULTS_DIR + SCBOW_RESULTS + EXP_EXT + IMG_EXT)
saveClassifiedSentences(sent_X, scbow_cosSims, scbow_pred, RESULTS_DIR + SCBOW_RESULTS + EXP_EXT + FILE_EXT)

print("<================> Skip - Thoughts <===============>")
# - Calculate Cosine Similarity
skip_cosSims = UnsupervisedMetrics.CosineSimilarity(skip_X, skip_X_VNC)

# - Get Predictions
skip_pred = UnsupervisedMetrics.ThresholdClassifier(skip_cosSims, T=COS_DIST_T, Op=COS_DIST_Op)

# Display Classifications:
if(SAVE_PLT): gen_plot(skip_X, y, skip_pred, "Original Skip-Thoughts Labels", "Cosine Similarity Labels", RESULTS_DIR + SKIP_RESULTS + EXP_EXT + IMG_EXT)
saveClassifiedSentences(sent_X, skip_cosSims, skip_pred, RESULTS_DIR + SKIP_RESULTS + EXP_EXT + FILE_EXT)

print("<=====================> ELMo <=====================>")
# - Calculate Cosine Similarity
elmo_cosSims = UnsupervisedMetrics.CosineSimilarity(elmo_X, elmo_X_VNC)

# - Get Predictions
elmo_pred = UnsupervisedMetrics.ThresholdClassifier(elmo_cosSims, T=COS_DIST_T, Op=COS_DIST_Op)

# Display Classifications:
if(SAVE_PLT): gen_plot(elmo_X, y, elmo_pred, "Original ELMo Labels", "Cosine Similarity Labels", RESULTS_DIR + ELMO_RESULTS + EXP_EXT + IMG_EXT)
saveClassifiedSentences(sent_X, elmo_cosSims, elmo_pred, RESULTS_DIR + ELMO_RESULTS + EXP_EXT + FILE_EXT)
