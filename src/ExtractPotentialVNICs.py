
import os
import re
import sys
import pickle
import argparse
import numpy as np
import pandas as pd

import progressbar

from SynLexFixedness import SynLexFixedness

# ------------- ARGS ------------- #
parser = argparse.ArgumentParser()

parser.add_argument("--VNIC_DIR_PMI" , "--PMI_candidates_out_dir" , type=str, help="Output File for the VNIC Candidates based on the PMI Metric.")
parser.add_argument("--VNIC_DIR_LEX" , "--LEX_candidates_out_dir" , type=str, help="Output File for the VNIC Candidates based on the Lexical Fixedness Metric.")
parser.add_argument("--VNIC_DIR_SYN" , "--SYN_candidates_out_dir" , type=str, help="Output File for the VNIC Candidates based on the Syntactical Fixedness Metric.")
parser.add_argument("--VNIC_DIR_OVA" , "--OVA_candidates_out_dir" , type=str, help="Output File for the VNIC Candidates based on the Overall Fixedness Metric.")
parser.add_argument("--PAT_MODEL" , "--pattern_counts_model"      , type=str, help="Location of the Pattern Counts Model.")
parser.add_argument("--W2V_MODEL" , "--Word2Vec_model"            , type=str, help="Location of the Word2Vec Model.")

parser.add_argument("--FREQ_T" , "--frequency_threshold" , type=int, help="Minimum number of instances requiered to take the VNIC Candidate into account.")

parser.add_argument("--K"      , "--lexical_fixedness_k"    , type=int  , help="K Parameters for Number of Similar Verb/Nouns for Lexical Fixedness Calculation.")
parser.add_argument("--ALPHA"  , "--overall_fixedness_alpha", type=float, help="ALPHA Parameter for Overall Fixedness Calculation.")
parser.add_argument("--USE_LIN", help="Use flag to indicate Lin's Thesaurus for Similar Verb/Nouns. Defaults to Word2Vec and WordNet.", action="store_true")

args = parser.parse_args()
# ------------- ARGS ------------- #

VNIC_DIR_PMI  = "./VNICs/PotentialVNICs_PMI.csv"
VNIC_DIR_LEX  = "./VNICs/PotentialVNICs_LEX.csv"
VNIC_DIR_SYN  = "./VNICs/PotentialVNICs_SYN.csv"
VNIC_DIR_OVA  = "./VNICs/PotentialVNICs_OVA.csv"
PAT_MODEL     = "./PatternCounts/PatternCounts_130619.pickle"
W2V_MODEL     = "./Word2Vec/models/W2V_ver1_lemm.model"

# Frequency Threshold for Instances
FREQ_T = 150

# Syntactical and Lexical Fixedness Parameters
K        = 50
ALPHA    = 0.6
LOG_BASE = 2
USE_LIN  = False

def main():
    # Initialize SynLexFixedness
    pat_SynLexFix = SynLexFixedness(modelDir=PAT_MODEL, w2vModelDir=W2V_MODEL, K=K)
    vncs_n = len(pat_SynLexFix.model)

    verb_np = np.full(vncs_n, 'generic_verb_instance')
    noun_np = np.full(vncs_n, 'generic_noun_instance')
    pmis_np = np.full(vncs_n, -1000.0)
    lexF_np = np.full(vncs_n, -1000.0)
    synF_np = np.full(vncs_n, -1000.0)
    ovaF_np = np.full(vncs_n, -1000.0)
    freq_np = np.zeros(vncs_n)

    # Iterate over all VNCs
    it = 0
    prog = progressbar.ProgressBar(max_value=vncs_n)
    prog.start()
    for vnc in pat_SynLexFix.model:
        verb_np[it] = vnc[0]
        noun_np[it] = vnc[1]
        if(pat_SynLexFix.model[vnc][0] >= FREQ_T):
            pmis_np[it] = pat_SynLexFix.PMI(vnc[0], vnc[1])
            lexF_np[it] = pat_SynLexFix.Fixedness_Lex(vnc[0], vnc[1], vK=K, nK=K, logBase=LOG_BASE, useLin=USE_LIN)
            synF_np[it] = pat_SynLexFix.Fixedness_Syn(vnc[0], vnc[1], logBase=LOG_BASE)
            ovaF_np[it] = pat_SynLexFix.Fixedness_Overall(vnc[0], vnc[1], alpha=ALPHA, vK=K, nK=K, logBase=LOG_BASE, useLin=USE_LIN)
        freq_np[it] = pat_SynLexFix.model[vnc][0]

        it += 1

        prog.update(it)

    vnics = pd.DataFrame(index=range(vncs_n), data={'Verb': verb_np, 'Noun': noun_np, 'PMI': pmis_np, 'Lex Fix': lexF_np, 'Syn Fix': synF_np, 'Ova Fix': ovaF_np, 'Freq': freq_np})

    vnics.sort_values(['PMI'], inplace=True, ascending=False)
    vnics.to_csv(path_or_buf=VNIC_DIR_PMI, sep=',')

    vnics.sort_values(['Lex Fix'], inplace=True, ascending=False)
    vnics.to_csv(path_or_buf=VNIC_DIR_LEX, sep=',')

    vnics.sort_values(['Syn Fix'], inplace=True, ascending=False)
    vnics.to_csv(path_or_buf=VNIC_DIR_SYN, sep=',')

    vnics.sort_values(['Ova Fix'], inplace=True, ascending=False)
    vnics.to_csv(path_or_buf=VNIC_DIR_OVA, sep=',')

if __name__ == '__main__':








    if(args.VNIC_DIR_PMI):
        VNIC_DIR_PMI = args.VNIC_DIR_PMI
    if(args.VNIC_DIR_LEX):
        VNIC_DIR_LEX = args.VNIC_DIR_LEX
    if(args.VNIC_DIR_SYN):
        VNIC_DIR_SYN = args.VNIC_DIR_SYN
    if(args.VNIC_DIR_OVA):
        VNIC_DIR_OVA = args.VNIC_DIR_OVA
    if(args.PAT_MODEL):
        PAT_MODEL = args.PAT_MODEL
    if(args.W2V_MODEL):
        W2V_MODEL = args.W2V_MODEL

    if(args.FREQ_T):
        FREQ_T = args.FREQ_T

    if(args.K):
        K = args.K
    if(args.ALPHA):
        ALPHA = args.ALPHA
    if(args.USE_LIN):
        USE_LIN = True
        W2V_MODEL = None

    main()

