"""
    File:   Experiment
    Author: Jose Juan Zavala Iglesias
    Date:   16/06/2019

    DBSCAN Classification evaluation for VNC Tokens Dataset using word embeddings generated with:
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
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
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
W2V_RESULTS   = "W2V"
SCBOW_RESULTS = "SCBOW"
SKIP_RESULTS  = "SKIP"
ELMO_RESULTS  = "ELMO"
IDIOMATIC_EXT = "_i"
LITERAL_EXT   = "_l"
TRAIN_EXT     = "_train"
PRED_EXT      = "_pred"
EXP_EXT       = "_DBSCAN_fixedness_cform"
FILE_EXT      = ".csv"
IMG_EXT       = ".png"

# Experiment Dirs
RESULTS_DIR = "./results/Experiment_2_6/"

# k-Means Parameters:
N_CLUSTERS = 2
N_INIT     = 10
MAX_ITER   = 300
N_JOBS     = -1
VERBOSE    = 1

# Split Parameters
RND_STATE = 10
TEST_SIZE = 0.3
SHUFFLE   = True

# Cluster Initialization
RND_SEED = 10

def gen_plot(feat, targ, pred, title_targ, title_pred, saveDir):
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
    plt.show()

def saveClassifiedSentences(all_sent, all_targ, all_pred, fileDir):
    for sent, targ, pred in zip(all_sent, all_targ, all_pred):
        if(targ != pred):
            print("Sentence", sent, "| Target:", targ, "| Clustered As:", pred)

    all_sent = all_sent.reshape((all_sent.size, 1))
    all_targ = all_targ.reshape((all_targ.size, 1))
    all_pred = all_pred.reshape((all_pred.size, 1))
    data = np.append(all_sent, all_targ, axis=1)
    data = np.append(data, all_pred, axis = 1)
    pd.DataFrame(data = data, columns=['Sentence', 'Target', 'Prediction']).to_csv(fileDir, sep='\t')

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

# Original Sentences
og_sent = np.genfromtxt(OG_SENT_DIR, delimiter="\t", dtype=None, encoding="utf_8")[indexes]

# Sentence Embeddings
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

og_sent_train, og_sent_test, _, _ = train_test_split(og_sent, targets_idiomatic, test_size=TEST_SIZE, shuffle=SHUFFLE, random_state=RND_STATE)

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
w2v_pred_train = DBSCAN(n_jobs=N_JOBS).fit(w2v_X_train)

# Display Clusters:
w2v_clust_labels = (np.array(w2v_pred_train.labels_) == w2v_y_train[cent_i])
gen_plot(w2v_X_train, w2v_y_train, w2v_clust_labels, "Original W2V Labels", "DBSCAN Labels", RESULTS_DIR + W2V_RESULTS + EXP_EXT + TRAIN_EXT + IMG_EXT)
saveClassifiedSentences(og_sent_train, w2v_y_train, w2v_clust_labels, RESULTS_DIR + W2V_RESULTS + PRED_EXT + EXP_EXT + FILE_EXT)

print("Results:", classification_report(w2v_y_train, w2v_clust_labels))

print("<=================> Siamese CBOW <=================>")
# - Run KMeans
scbow_pred_train = DBSCAN(n_jobs=N_JOBS).fit(scbow_X_train)

# Display Clusters:
scbow_clust_labels = (np.array(scbow_pred_train.labels_) == scbow_y_train[cent_i])
gen_plot(scbow_X_train, scbow_y_train, scbow_clust_labels, "Original Siamese CBOW Labels", "DBSCAN Labels", RESULTS_DIR + SCBOW_RESULTS + EXP_EXT + TRAIN_EXT + IMG_EXT)
saveClassifiedSentences(og_sent_train, scbow_y_train, scbow_clust_labels, RESULTS_DIR + SCBOW_RESULTS + PRED_EXT + EXP_EXT + FILE_EXT)

print("Results:", classification_report(scbow_y_train, scbow_clust_labels))

print("<================> Skip - Thoughts <===============>")
# - Run KMeans
skip_pred_train = DBSCAN(n_jobs=N_JOBS).fit(skip_X_train)

# Display Clusters:
skip_clust_labels = (np.array(skip_pred_train.labels_) == skip_y_train[cent_i])
gen_plot(skip_X_train, skip_y_train, skip_clust_labels, "Original Siamese Skip-Thoughts Labels", "DBSCAN Labels", RESULTS_DIR + SKIP_RESULTS + EXP_EXT + TRAIN_EXT + IMG_EXT)
saveClassifiedSentences(og_sent_train, skip_y_train, skip_clust_labels, RESULTS_DIR + SKIP_RESULTS + PRED_EXT + EXP_EXT + FILE_EXT)

print("Results:", classification_report(skip_y_train, skip_clust_labels))

print("<=====================> ELMo <=====================>")
# - Run KMeans
elmo_pred_train = DBSCAN(n_jobs=N_JOBS).fit(elmo_X_train)

# Display Clusters:
elmo_clust_labels = (np.array(elmo_pred_train.labels_) == elmo_y_train[cent_i])
gen_plot(elmo_X_train, elmo_y_train, elmo_clust_labels, "Original Siamese ELMo Labels", "DBSCAN Labels", RESULTS_DIR + ELMO_RESULTS + EXP_EXT + TRAIN_EXT + IMG_EXT)
saveClassifiedSentences(og_sent_train, elmo_y_train, elmo_clust_labels, RESULTS_DIR + ELMO_RESULTS + PRED_EXT + EXP_EXT + FILE_EXT)

print("Results:", classification_report(elmo_y_train, elmo_clust_labels))
