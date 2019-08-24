"""
    File:   Experiment_2_1
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
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.utils import shuffle
from sklearn.utils import resample
from sklearn.decomposition import PCA

# ------------- ARGS ------------- #
parser = argparse.ArgumentParser()

parser.add_argument("--OG_SENT_DIR"   , "--original_sentences_dir"            , type=str, help="Location of the File Containing the Original Extracted Sentences.")
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

parser.add_argument("--RESULTS_DIR" , "--results_directory" , type=str, help="Results Directory.")
parser.add_argument("--EXP_EXT"     , "--experiment_suffix" , type=str, help="Experiments Name Extension.")

parser.add_argument("--USE_CFORM"   , help="Use flag to indicate if CForm Feature Should Be Added."               , action="store_true")
parser.add_argument("--USE_SYN_FIX" , help="Use flag to indicate if Syntactic Fixedness Feature Should Be Added." , action="store_true")
parser.add_argument("--USE_LEX_FIX" , help="Use flag to indicate if Lexical Fixedness Feature Should Be Added."   , action="store_true")
parser.add_argument("--USE_OVA_FIX" , help="Use flag to indicate if Overall Fixedness Feature Should Be Added."   , action="store_true")

parser.add_argument("--N_CLUSTERS" , "--num_clusters"       , type=int, help="k Number of Clusters.")
parser.add_argument("--N_INIT"     , "--num_runs"           , type=int, help="Number of Runs with Different Centroid Seeds.")
parser.add_argument("--MAX_ITER"   , "--maximum_iterations" , type=int, help="Maximum number of iterations of the k-means algorithm for a single run.")
parser.add_argument("--N_JOBS"     , "--number_of_jobs"     , type=int, help="Number of Jobs to Use for Computation.")
parser.add_argument("--VERBOSE"    , "--verbose_level"      , type=int, help="Verbosity Mode.")

parser.add_argument("--RND_STATE" , "--state_seed"       , type=int, help="Random Seed for Replication Purposes.")
parser.add_argument("--SEED_SIZE" , "--seed_sample_size" , type=int, help="Seed Sample Size for k-Means Cluster Class.")

parser.add_argument("--SAVE_PLT" , help="Use flag to indicate if Plots Should be Saved.", action="store_true")

args = parser.parse_args()
# ------------- ARGS ------------- #

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

# Experiment Dirs
RESULTS_DIR = "./results/Experiment_2_1/"
EXP_EXT     = "_clust"

# Features:
USE_CFORM   = False
USE_SYN_FIX = False
USE_LEX_FIX = False
USE_OVA_FIX = False

# File Extensions
FILE_EXT = ".tsv"
CSV_EXT  = ".csv"
IMG_EXT  = ".png"

# k-Means Parameters:
N_CLUSTERS = 2
N_INIT     = 10
MAX_ITER   = 300
N_JOBS     = -1
VERBOSE    = 1

# Shuffle Parameters
RND_STATE = 42
SEED_SIZE = 1000

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

def getPredictions(labels, pred_Seed, y_Seed, k):
    pred = np.zeros((labels.shape[0],), dtype=bool)

    for cluster in range(k):
        idx_clst = np.where(labels == cluster)[0]
        idx_seed = np.where(pred_Seed == cluster)[0]

        if(len(idx_seed) > 0):
            trueCnt = len(np.where(y_Seed[idx_seed] == True)[0])

            if(trueCnt > (len(idx_seed) - trueCnt)):
                pred[idx_clst] = True
            else:
                pred[idx_clst] = False
        else:
            # If no example from the seed is in the cluster, predict randomly
            pred[idx_clst] = random.choice([True, False])

    return pred


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

    # Shuffle Sets:
    sent_X, w2v_X, scbow_X, skip_X, elmo_X, y = shuffle(og_sent, features_w2v, features_scbow, features_skip, features_elmo, targets_idiomatic, random_state=RND_STATE)

    # Get Random Seed Sample:
    w2v_X_Seed, scbow_X_Seed, skip_X_Seed, elmo_X_Seed, y_Seed = resample(features_w2v, features_scbow, features_skip, features_elmo, targets_idiomatic, n_samples=SEED_SIZE, random_state=RND_STATE, replace=False, stratify=targets_idiomatic)

    # -- Train k-Means -- #

    print("<===================> Word2Vec <===================>")
    # - Run KMeans
    w2v_kMeans = KMeans(n_clusters=N_CLUSTERS, init="k-means++", n_init=N_INIT, n_jobs=N_JOBS, max_iter=MAX_ITER, verbose=VERBOSE, random_state=RND_STATE).fit(w2v_X)

    # - Get Seed Classifications
    w2v_seed_labels = w2v_kMeans.predict(w2v_X_Seed)

    # Display Clusters:
    w2v_clust_labels = getPredictions(w2v_kMeans.labels_, w2v_seed_labels, y_Seed, N_CLUSTERS)
    if(SAVE_PLT): gen_plot(w2v_X, y, w2v_clust_labels, "Original Word2Vec Labels", "k-Means Labels", RESULTS_DIR + W2V_RESULTS + EXP_EXT + IMG_EXT)
    saveClassifiedSentences(sent_X, y, w2v_clust_labels, RESULTS_DIR + W2V_RESULTS + EXP_EXT + FILE_EXT)

    print("Results:", classification_report(y, w2v_clust_labels))
    results_w2v = pd.DataFrame.from_dict(classification_report(y, w2v_clust_labels, output_dict=True))
    results_w2v.to_csv(RESULTS_DIR + W2V_RESULTS + EXP_EXT  + CSV_EXT)

    print("<=================> Siamese CBOW <=================>")
    # - Run KMeans
    scbow_kMeans = KMeans(n_clusters=N_CLUSTERS, init="k-means++", n_init=N_INIT, n_jobs=N_JOBS, max_iter=MAX_ITER, verbose=VERBOSE, random_state=RND_STATE).fit(scbow_X)

    # - Get Seed Classifications
    scbow_seed_labels = scbow_kMeans.predict(scbow_X_Seed)

    # Display Clusters:
    scbow_clust_labels = getPredictions(scbow_kMeans.labels_, scbow_seed_labels, y_Seed, N_CLUSTERS)
    if(SAVE_PLT): gen_plot(scbow_X, y, scbow_clust_labels, "Original Siamese CBOW Labels", "k-Means Labels", RESULTS_DIR + SCBOW_RESULTS + EXP_EXT + IMG_EXT)
    saveClassifiedSentences(sent_X, y, scbow_clust_labels, RESULTS_DIR + SCBOW_RESULTS + EXP_EXT + FILE_EXT)

    print("Results:", classification_report(y, scbow_clust_labels))
    results_scbow = pd.DataFrame.from_dict(classification_report(y, scbow_clust_labels, output_dict=True))
    results_scbow.to_csv(RESULTS_DIR + SCBOW_RESULTS + EXP_EXT  + CSV_EXT)

    print("<================> Skip - Thoughts <===============>")
    # - Run KMeans
    skip_kMeans = KMeans(n_clusters=N_CLUSTERS, init="k-means++", n_init=N_INIT, n_jobs=N_JOBS, max_iter=MAX_ITER, verbose=VERBOSE, random_state=RND_STATE).fit(skip_X)

    # - Get Seed Classifications
    skip_seed_labels = skip_kMeans.predict(skip_X_Seed)

    # Display Clusters:
    skip_clust_labels = getPredictions(skip_kMeans.labels_, skip_seed_labels, y_Seed, N_CLUSTERS)
    if(SAVE_PLT): gen_plot(skip_X, y, skip_clust_labels, "Original Skip-Thoughts Labels", "k-Means Labels", RESULTS_DIR + SKIP_RESULTS + EXP_EXT + IMG_EXT)
    saveClassifiedSentences(sent_X, y, skip_clust_labels, RESULTS_DIR + SKIP_RESULTS + EXP_EXT + FILE_EXT)

    print("Results:", classification_report(y, skip_clust_labels))
    results_skip = pd.DataFrame.from_dict(classification_report(y, skip_clust_labels, output_dict=True))
    results_skip.to_csv(RESULTS_DIR + SKIP_RESULTS + EXP_EXT  + CSV_EXT)

    print("<=====================> ELMo <=====================>")
    # - Run KMeans
    elmo_kMeans = KMeans(n_clusters=N_CLUSTERS, init="k-means++", n_init=N_INIT, n_jobs=N_JOBS, max_iter=MAX_ITER, verbose=VERBOSE, random_state=RND_STATE).fit(elmo_X)

    # - Get Seed Classifications
    elmo_seed_labels = elmo_kMeans.predict(elmo_X_Seed)

    # Display Clusters:
    elmo_clust_labels = getPredictions(elmo_kMeans.labels_, elmo_seed_labels, y_Seed, N_CLUSTERS)
    if(SAVE_PLT): gen_plot(elmo_X, y, elmo_clust_labels, "Original ELMo Labels", "k-Means Labels", RESULTS_DIR + ELMO_RESULTS + EXP_EXT + IMG_EXT)
    saveClassifiedSentences(sent_X, y, elmo_clust_labels, RESULTS_DIR + ELMO_RESULTS + EXP_EXT + FILE_EXT)

    print("Results:", classification_report(y, elmo_clust_labels))
    results_elmo = pd.DataFrame.from_dict(classification_report(y, elmo_clust_labels, output_dict=True))
    results_elmo.to_csv(RESULTS_DIR + ELMO_RESULTS + EXP_EXT  + CSV_EXT)

if __name__ == '__main__':

    if(args.OG_SENT_DIR):
        OG_SENT_DIR = args.OG_SENT_DIR
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

    if(args.N_CLUSTERS):
        N_CLUSTERS = args.N_CLUSTERS
    if(args.N_INIT):
        N_INIT = args.N_INIT
    if(args.MAX_ITER):
        MAX_ITER = args.MAX_ITER
    if(args.N_JOBS):
        N_JOBS = args.N_JOBS
    if(args.VERBOSE):
        VERBOSE = args.VERBOSE

    if(args.RND_STATE):
        RND_STATE = args.RND_STATE
    if(args.SEED_SIZE):
        SEED_SIZE = args.SEED_SIZE

    if(args.SAVE_PLT):
        SAVE_PLT = True

    main()
