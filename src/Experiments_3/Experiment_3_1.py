"""
    File:   Experiment_3_1
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
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

import UnsupervisedMetrics

# ------------- ARGS ------------- #
parser = argparse.ArgumentParser()

parser.add_argument("--OG_SENT_DIR"      , "--original_sentences_dir"            , type=str, help="Location of the File Containing the Original Extracted Sentences.")
parser.add_argument("--TARGETS_DIR"      , "--targets_directory"                 , type=str, help="Location of the File Containing the Targets in VNC-Token format.")
parser.add_argument("--W2V_DIR"          , "--w2v_directory"                     , type=str, help="Location of the Input Dir Containing the Word2Vec Embeddings.")
parser.add_argument("--SCBOW_DIR"        , "--scbow_directory"                   , type=str, help="Location of the Input Dir Containing the Siamese SCBOW Embeddings.")
parser.add_argument("--SKIP_DIR"         , "--skip-thoughts_directory"           , type=str, help="Location of the Input Dir Containing the Skip-Thoughts Embeddings.")
parser.add_argument("--ELMO_DIR"         , "--elmo_directory"                    , type=str, help="Location of the Input Dir Containing the ELMo Embeddings.")
parser.add_argument("--VECTORS_FILE"     , "--embedded_vectors_file"             , type=str, help="Name of the Embeddings File.")
parser.add_argument("--VECTORS_FILE_VNC" , "--embedded_vnc_vectors_file"         , type=str, help="Name of the VNC Embeddings File.")
parser.add_argument("--W2V_RESULTS"      , "--w2v_results_file_prefix"           , type=str, help="Location of the Output File Containing the Cross-Validation Results for Word2Vec's SVM.")
parser.add_argument("--SCBOW_RESULTS"    , "--scbow_results_file_prefix"         , type=str, help="Location of the Output File Containing the Cross-Validation Results for Siamese CBOW's SVM.")
parser.add_argument("--SKIP_RESULTS"     , "--skip-thoughts_results_file_prefix" , type=str, help="Location of the Output File Containing the Cross-Validation Results for Skip-Thoughts's SVM.")
parser.add_argument("--ELMO_RESULTS"     , "--elmo_results_file_prefix"          , type=str, help="Location of the Output File Containing the Cross-Validation Results for ELMo's SVM.")

parser.add_argument("--RESULTS_DIR" , "--results_directory" , type=str, help="Results Directory.")
parser.add_argument("--EXP_EXT"     , "--experiment_suffix" , type=str, help="Experiments Name Extension.")

parser.add_argument("--COS_DIST_Op" , "--threshold_operator"          , type=str, help="Operator for Positive Classification.")

parser.add_argument("--SAVE_PLT" , help="Use flag to indicate if Plots Should be Saved.", action="store_true")

args = parser.parse_args()
# ------------- ARGS ------------- #

# Files and Directories:
OG_SENT_DIR      = "../targets/Extracted_Sentences.txt"
TARGETS_DIR      = "../targets/English_VNC_Cook/VNC-Tokens_cleaned"
W2V_DIR          = "../Word2Vec/"
SCBOW_DIR        = "../SiameseCBOW/"
SKIP_DIR         = "../SkipThoughts/"
ELMO_DIR         = "../ELMo/"
VECTORS_FILE     = "embeddings.csv"
VECTORS_FILE_VNC = "embeddings_VNC.csv"
W2V_RESULTS      = "W2V"
SCBOW_RESULTS    = "SCBOW"
SKIP_RESULTS     = "SKIP"
ELMO_RESULTS     = "ELMO"

# Experiment Dirs
RESULTS_DIR = "./results/Experiment_2_4/"
EXP_EXT     = "_cosineSimilarity"

# File Extensions
FILE_EXT = ".tsv"
CSV_EXT  = ".csv"
IMG_EXT  = ".png"

# Unsupervised Parameters:
COS_DIST_T  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
COS_DIST_Op = '<'

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

    # VNC Embeddings
    features_w2v_VNC   = np.genfromtxt(W2V_DIR   + VECTORS_FILE_VNC, delimiter=',')[indexes]
    features_scbow_VNC = np.genfromtxt(SCBOW_DIR + VECTORS_FILE_VNC, delimiter=',')[indexes]
    features_skip_VNC  = np.genfromtxt(SKIP_DIR  + VECTORS_FILE_VNC, delimiter=',')[indexes]
    features_elmo_VNC  = np.genfromtxt(ELMO_DIR  + VECTORS_FILE_VNC, delimiter=',')[indexes]

    # Initialize Results DataFrames
    df_w2v_accuracy  = pd.DataFrame(index=COS_DIST_T, columns=["Threshold"])
    df_w2v_f1_score  = pd.DataFrame(index=COS_DIST_T, columns=["Threshold"])
    df_w2v_precision = pd.DataFrame(index=COS_DIST_T, columns=["Threshold"])
    df_w2v_recall    = pd.DataFrame(index=COS_DIST_T, columns=["Threshold"])

    df_scbow_accuracy  = pd.DataFrame(index=COS_DIST_T, columns=["Threshold"])
    df_scbow_f1_score  = pd.DataFrame(index=COS_DIST_T, columns=["Threshold"])
    df_scbow_precision = pd.DataFrame(index=COS_DIST_T, columns=["Threshold"])
    df_scbow_recall    = pd.DataFrame(index=COS_DIST_T, columns=["Threshold"])

    df_skip_accuracy  = pd.DataFrame(index=COS_DIST_T, columns=["Threshold"])
    df_skip_f1_score  = pd.DataFrame(index=COS_DIST_T, columns=["Threshold"])
    df_skip_precision = pd.DataFrame(index=COS_DIST_T, columns=["Threshold"])
    df_skip_recall    = pd.DataFrame(index=COS_DIST_T, columns=["Threshold"])

    df_elmo_accuracy  = pd.DataFrame(index=COS_DIST_T, columns=["Threshold"])
    df_elmo_f1_score  = pd.DataFrame(index=COS_DIST_T, columns=["Threshold"])
    df_elmo_precision = pd.DataFrame(index=COS_DIST_T, columns=["Threshold"])
    df_elmo_recall    = pd.DataFrame(index=COS_DIST_T, columns=["Threshold"])

    # Shuffle Sets:
    sent_X, w2v_X, w2v_X_VNC, scbow_X, scbow_X_VNC, skip_X, skip_X_VNC, elmo_X, elmo_X_VNC, y = og_sent, features_w2v, features_w2v_VNC, features_scbow, features_scbow_VNC, features_skip, features_skip_VNC, features_elmo, features_elmo_VNC, targets_idiomatic

    for cos_dist_t in COS_DIST_T:
        exp_ext = EXP_EXT + "_COS_DIST_T_" + str(cos_dist_t)

        print("<===================> Word2Vec <===================>")
        # - Calculate Cosine Similarity
        w2v_cosSims = UnsupervisedMetrics.CosineSimilarity(w2v_X, w2v_X_VNC)

        # - Get Predictions
        w2v_pred = UnsupervisedMetrics.ThresholdClassifier(w2v_cosSims, T=cos_dist_t, Op=COS_DIST_Op)

        # Display Classifications:
        if(SAVE_PLT): gen_plot(w2v_X, y, w2v_pred, "Original Word2Vec Labels", "Cosine Similarity Labels", RESULTS_DIR + W2V_RESULTS + exp_ext + IMG_EXT)
        saveClassifiedSentences(sent_X, w2v_cosSims, y, w2v_pred, RESULTS_DIR + W2V_RESULTS + exp_ext + FILE_EXT)

        print("Results:", classification_report(y, w2v_pred))
        results_w2v = pd.DataFrame.from_dict(classification_report(y, w2v_pred, output_dict=True))
        results_w2v.to_csv(RESULTS_DIR + W2V_RESULTS + exp_ext + CSV_EXT)

        df_w2v_accuracy[cos_dist_t, "Threshold"]  = accuracy_score(y, w2v_pred)
        df_w2v_f1_score[cos_dist_t, "Threshold"]  = results_w2v['True']['f1-score']
        df_w2v_precision[cos_dist_t, "Threshold"] = results_w2v['True']['precision']
        df_w2v_recall[cos_dist_t, "Threshold"]    = results_w2v['True']['recall']

        print("<=================> Siamese CBOW <=================>")
        # - Calculate Cosine Similarity
        scbow_cosSims = UnsupervisedMetrics.CosineSimilarity(scbow_X, scbow_X_VNC)

        # - Get Predictions
        scbow_pred = UnsupervisedMetrics.ThresholdClassifier(scbow_cosSims, T=cos_dist_t, Op=COS_DIST_Op)

        # Display Classifications:
        if(SAVE_PLT): gen_plot(scbow_X, y, scbow_pred, "Original Siamese CBOW Labels", "Cosine Similarity Labels", RESULTS_DIR + SCBOW_RESULTS + exp_ext + IMG_EXT)
        saveClassifiedSentences(sent_X, scbow_cosSims, y, scbow_pred, RESULTS_DIR + SCBOW_RESULTS + exp_ext + FILE_EXT)

        print("Results:", classification_report(y, scbow_pred))
        results_scbow = pd.DataFrame.from_dict(classification_report(y, scbow_pred, output_dict=True))
        results_scbow.to_csv(RESULTS_DIR + SCBOW_RESULTS + exp_ext + CSV_EXT)

        df_scbow_accuracy[cos_dist_t, "Threshold"]  = accuracy_score(y, scbow_pred)
        df_scbow_f1_score[cos_dist_t, "Threshold"]  = results_scbow['True']['f1-score']
        df_scbow_precision[cos_dist_t, "Threshold"] = results_scbow['True']['precision']
        df_scbow_recall[cos_dist_t, "Threshold"]    = results_scbow['True']['recall']

        print("<================> Skip - Thoughts <===============>")
        # - Calculate Cosine Similarity
        skip_cosSims = UnsupervisedMetrics.CosineSimilarity(skip_X, skip_X_VNC)

        # - Get Predictions
        skip_pred = UnsupervisedMetrics.ThresholdClassifier(skip_cosSims, T=cos_dist_t, Op=COS_DIST_Op)

        # Display Classifications:
        if(SAVE_PLT): gen_plot(skip_X, y, skip_pred, "Original Skip-Thoughts Labels", "Cosine Similarity Labels", RESULTS_DIR + SKIP_RESULTS + exp_ext + IMG_EXT)
        saveClassifiedSentences(sent_X, skip_cosSims, y, skip_pred, RESULTS_DIR + SKIP_RESULTS + exp_ext + FILE_EXT)

        print("Results:", classification_report(y, skip_pred))
        results_skip = pd.DataFrame.from_dict(classification_report(y, skip_pred, output_dict=True))
        results_skip.to_csv(RESULTS_DIR + SKIP_RESULTS + exp_ext + CSV_EXT)

        df_skip_accuracy[cos_dist_t, "Threshold"]  = accuracy_score(y, skip_pred)
        df_skip_f1_score[cos_dist_t, "Threshold"]  = results_skip['True']['f1-score']
        df_skip_precision[cos_dist_t, "Threshold"] = results_skip['True']['precision']
        df_skip_recall[cos_dist_t, "Threshold"]    = results_skip['True']['recall']

        print("<=====================> ELMo <=====================>")
        # - Calculate Cosine Similarity
        elmo_cosSims = UnsupervisedMetrics.CosineSimilarity(elmo_X, elmo_X_VNC)

        # - Get Predictions
        elmo_pred = UnsupervisedMetrics.ThresholdClassifier(elmo_cosSims, T=cos_dist_t, Op=COS_DIST_Op)

        # Display Classifications:
        if(SAVE_PLT): gen_plot(elmo_X, y, elmo_pred, "Original ELMo Labels", "Cosine Similarity Labels", RESULTS_DIR + ELMO_RESULTS + exp_ext + IMG_EXT)
        saveClassifiedSentences(sent_X, elmo_cosSims, y, elmo_pred, RESULTS_DIR + ELMO_RESULTS + exp_ext + FILE_EXT)

        print("Results:", classification_report(y, elmo_pred))
        results_elmo = pd.DataFrame.from_dict(classification_report(y, elmo_pred, output_dict=True))
        results_elmo.to_csv(RESULTS_DIR + ELMO_RESULTS + exp_ext + CSV_EXT)

        df_elmo_accuracy[cos_dist_t, "Threshold"]  = accuracy_score(y, elmo_pred)
        df_elmo_f1_score[cos_dist_t, "Threshold"]  = results_elmo['True']['f1-score']
        df_elmo_precision[cos_dist_t, "Threshold"] = results_elmo['True']['precision']
        df_elmo_recall[cos_dist_t, "Threshold"]    = results_elmo['True']['recall']

    # Store Result Grid
    df_w2v_accuracy.to_csv(RESULTS_DIR  + W2V_RESULTS + "_ACCURACY_"  + EXP_EXT + CSV_EXT)  
    df_w2v_f1_score.to_csv(RESULTS_DIR  + W2V_RESULTS + "_F1-SCORE_"  + EXP_EXT + CSV_EXT)  
    df_w2v_precision.to_csv(RESULTS_DIR + W2V_RESULTS + "_PREVISION_" + EXP_EXT + CSV_EXT) 
    df_w2v_recall.to_csv(RESULTS_DIR    + W2V_RESULTS + "_RECALL_"    + EXP_EXT + CSV_EXT)    

    df_scbow_accuracy.to_csv(RESULTS_DIR  + SCBOW_RESULTS + "_ACCURACY_"  + EXP_EXT + CSV_EXT)  
    df_scbow_f1_score.to_csv(RESULTS_DIR  + SCBOW_RESULTS + "_F1-SCORE_"  + EXP_EXT + CSV_EXT)  
    df_scbow_precision.to_csv(RESULTS_DIR + SCBOW_RESULTS + "_PREVISION_" + EXP_EXT + CSV_EXT) 
    df_scbow_recall.to_csv(RESULTS_DIR    + SCBOW_RESULTS + "_RECALL_"    + EXP_EXT + CSV_EXT)    

    df_skip_accuracy.to_csv(RESULTS_DIR  + SKIP_RESULTS + "_ACCURACY_"  + EXP_EXT + CSV_EXT) 
    df_skip_f1_score.to_csv(RESULTS_DIR  + SKIP_RESULTS + "_F1-SCORE_"  + EXP_EXT + CSV_EXT) 
    df_skip_precision.to_csv(RESULTS_DIR + SKIP_RESULTS + "_PREVISION_" + EXP_EXT + CSV_EXT)
    df_skip_recall.to_csv(RESULTS_DIR    + SKIP_RESULTS + "_RECALL_"    + EXP_EXT + CSV_EXT)   

    df_elmo_accuracy.to_csv(RESULTS_DIR  + ELMO_RESULTS + "_ACCURACY_"  + EXP_EXT + CSV_EXT) 
    df_elmo_f1_score.to_csv(RESULTS_DIR  + ELMO_RESULTS + "_F1-SCORE_"  + EXP_EXT + CSV_EXT) 
    df_elmo_precision.to_csv(RESULTS_DIR + ELMO_RESULTS + "_PREVISION_" + EXP_EXT + CSV_EXT)
    df_elmo_recall.to_csv(RESULTS_DIR    + ELMO_RESULTS + "_RECALL_"    + EXP_EXT + CSV_EXT)

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
    if(args.VECTORS_FILE):
        VECTORS_FILE = args.VECTORS_FILE
    if(args.VECTORS_FILE_VNC):
        VECTORS_FILE_VNC = args.VECTORS_FILE_VNC
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

    if(args.SAVE_PLT):
        SAVE_PLT = True

    if(args.COS_DIST_Op):
        COS_DIST_Op = args.COS_DIST_Op

    main()
