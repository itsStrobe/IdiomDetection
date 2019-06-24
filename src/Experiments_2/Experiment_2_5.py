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
OG_SENT_DIR      = "../targets/Extracted_Sentences.txt"
TARGETS_DIR      = "../targets/English_VNC_Cook/VNC-Tokens_cleaned"
W2V_DIR          = "../Word2Vec/"
SCBOW_DIR        = "../SiameseCBOW/"
SKIP_DIR         = "../SkipThoughts/"
ELMO_DIR         = "../ELMo/"
OVA_FIX_DIR      = "../targets/OvaFix.csv"
VECTORS_FILE     = "embeddings.csv"
VECTORS_FILE_VNC = "embeddings_VNC.csv"
RESULTS_DIR      = "./results/"
W2V_RESULTS      = "W2V"
SCBOW_RESULTS    = "SCBOW"
SKIP_RESULTS     = "SKIP"
ELMO_RESULTS     = "ELMO"

# Experiment Suffix
EXP_EXT = "_unnamedMetric"

# File Extensions
FILE_EXT = ".tsv"
CSV_EXT  = ".csv"
IMG_EXT  = ".png"

# Unsupervised Parameters:
UNM_MET_T  = 0.7
UNM_MET_Op = '>'
BETA       = 0.5

# Split Parameters
RND_STATE = 42
TEST_SIZE = 0.3
SHUFFLE   = True

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

def saveClassifiedSentences(all_sent, all_cSim, all_targ, all_pred, fileDir):

    all_sent = all_sent.reshape((all_sent.size, 1))
    all_cSim = all_cSim.reshape((all_cSim.size, 1))
    all_targ = all_targ.reshape((all_targ.size, 1))
    all_pred = all_pred.reshape((all_pred.size, 1))
    data = np.append(all_sent, all_cSim, axis=1)
    data = np.append(data, all_targ, axis = 1)
    data = np.append(data, all_pred, axis = 1)
    pd.DataFrame(data = data, columns=['Sentence', 'Cos Similarity', 'Target', 'Prediction']).to_csv(fileDir, sep='\t')

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

# VNC Embeddings
features_w2v_VNC   = np.genfromtxt(W2V_DIR   + VECTORS_FILE_VNC, delimiter=',')[indexes]
features_scbow_VNC = np.genfromtxt(SCBOW_DIR + VECTORS_FILE_VNC, delimiter=',')[indexes]
features_skip_VNC  = np.genfromtxt(SKIP_DIR  + VECTORS_FILE_VNC, delimiter=',')[indexes]
features_elmo_VNC  = np.genfromtxt(ELMO_DIR  + VECTORS_FILE_VNC, delimiter=',')[indexes]

# Pre-Calculated Overall Fixedness
features_OVA = np.genfromtxt(OVA_FIX_DIR, delimiter=',')[indexes]

# Split Sets:
sent_X, w2v_X, w2v_X_VNC, scbow_X, scbow_X_VNC, skip_X, skip_X_VNC, elmo_X, elmo_X_VNC, ova_X, y = shuffle(og_sent, features_w2v, features_w2v_VNC, features_scbow, features_scbow_VNC, features_skip, features_skip_VNC, features_elmo, features_elmo_VNC, features_OVA, targets_idiomatic, random_state=RND_STATE)


print("<===================> Word2Vec <===================>")
# - Calculate Cosine Similarity
w2v_cosSims = UnsupervisedMetrics.CosineSimilarity(w2v_X, w2v_X_VNC)

# - Calculate Unnamed Metric
w2v_unnamedMetric = UnsupervisedMetrics.UnnamedMetric(w2v_cosSims, ova_X, beta=BETA)

# - Get Predictions
w2v_pred = UnsupervisedMetrics.ThresholdClassifier(w2v_unnamedMetric, T=UNM_MET_T, Op=UNM_MET_Op)

# Display Classifications:
if(SAVE_PLT): gen_plot(w2v_X, y, w2v_pred, "Original Word2Vec Labels", "Cosine Similarity Labels", RESULTS_DIR + W2V_RESULTS + EXP_EXT + IMG_EXT)
saveClassifiedSentences(sent_X, w2v_unnamedMetric, y, w2v_pred, RESULTS_DIR + W2V_RESULTS + EXP_EXT + FILE_EXT)

print("Results:", classification_report(y, w2v_pred))
results_w2v = pd.DataFrame.from_dict(classification_report(y, w2v_pred, output_dict=True))
results_w2v.to_csv(RESULTS_DIR + W2V_RESULTS + EXP_EXT  + CSV_EXT)

print("<=================> Siamese CBOW <=================>")
# - Calculate Cosine Similarity
scbow_cosSims = UnsupervisedMetrics.CosineSimilarity(scbow_X, scbow_X_VNC)

# - Calculate Unnamed Metric
scbow_unnamedMetric = UnsupervisedMetrics.UnnamedMetric(scbow_cosSims, ova_X, beta=BETA)

# - Get Predictions
scbow_pred = UnsupervisedMetrics.ThresholdClassifier(scbow_unnamedMetric, T=UNM_MET_T, Op=UNM_MET_Op)

# Display Classifications:
if(SAVE_PLT): if(SAVE_PLT): gen_plot(scbow_X, y, scbow_pred, "Original Siamese CBOW Labels", "Cosine Similarity Labels", RESULTS_DIR + SCBOW_RESULTS + EXP_EXT + IMG_EXT)
saveClassifiedSentences(sent_X, scbow_unnamedMetric, y, scbow_pred, RESULTS_DIR + SCBOW_RESULTS + EXP_EXT + FILE_EXT)

print("Results:", classification_report(y, scbow_pred))
results_scbow = pd.DataFrame.from_dict(classification_report(y, scbow_pred, output_dict=True))
results_scbow.to_csv(RESULTS_DIR + SCBOW_RESULTS + EXP_EXT  + CSV_EXT)

print("<================> Skip - Thoughts <===============>")
# - Calculate Cosine Similarity
skip_cosSims = UnsupervisedMetrics.CosineSimilarity(skip_X, skip_X_VNC)

# - Calculate Unnamed Metric
skip_unnamedMetric = UnsupervisedMetrics.UnnamedMetric(skip_cosSims, ova_X, beta=BETA)

# - Get Predictions
skip_pred = UnsupervisedMetrics.ThresholdClassifier(skip_unnamedMetric, T=UNM_MET_T, Op=UNM_MET_Op)

# Display Classifications:
if(SAVE_PLT): gen_plot(skip_X, y, skip_pred, "Original Skip-Thoughts Labels", "Cosine Similarity Labels", RESULTS_DIR + SKIP_RESULTS + EXP_EXT + IMG_EXT)
saveClassifiedSentences(sent_X, skip_unnamedMetric, y, skip_pred, RESULTS_DIR + SKIP_RESULTS + EXP_EXT + FILE_EXT)

print("Results:", classification_report(y, skip_pred))
results_skip = pd.DataFrame.from_dict(classification_report(y, skip_pred, output_dict=True))
results_skip.to_csv(RESULTS_DIR + SKIP_RESULTS + EXP_EXT  + CSV_EXT)

print("<=====================> ELMo <=====================>")
# - Calculate Cosine Similarity
elmo_cosSims = UnsupervisedMetrics.CosineSimilarity(elmo_X, elmo_X_VNC)

# - Calculate Unnamed Metric
elmo_unnamedMetric = UnsupervisedMetrics.UnnamedMetric(elmo_cosSims, ova_X, beta=BETA)

# - Get Predictions
elmo_pred = UnsupervisedMetrics.ThresholdClassifier(elmo_unnamedMetric, T=UNM_MET_T, Op=UNM_MET_Op)

# Display Classifications:
if(SAVE_PLT): gen_plot(elmo_X, y, elmo_pred, "Original ELMo Labels", "Cosine Similarity Labels", RESULTS_DIR + ELMO_RESULTS + EXP_EXT + IMG_EXT)
saveClassifiedSentences(sent_X, elmo_unnamedMetric, y, elmo_pred, RESULTS_DIR + ELMO_RESULTS + EXP_EXT + FILE_EXT)

print("Results:", classification_report(y, elmo_pred))
results_elmo = pd.DataFrame.from_dict(classification_report(y, elmo_pred, output_dict=True))
results_elmo.to_csv(RESULTS_DIR + ELMO_RESULTS + EXP_EXT  + CSV_EXT)
