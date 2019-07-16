
import os
import re
import sys
import pickle
import numpy as np
import pandas as pd

import progressbar

from SynLexFixedness import SynLexFixedness

PAT_MODEL     = "./PatternCounts/PatternCounts_130619.pickle"
W2V_MODEL     = "./Word2Vec/models/W2V_ver1_lemm.model"
VNIC_DIR_PMI  = "./VNICs/PotentialVNICs_PMI.csv"
VNIC_DIR_LEX  = "./VNICs/PotentialVNICs_LEX.csv"
VNIC_DIR_SYN  = "./VNICs/PotentialVNICs_SYN.csv"
VNIC_DIR_OVA  = "./VNICs/PotentialVNICs_OVA.csv"

# Frequency Threshold for Instances
FREQ_T = 150

# Syntactical and Lexical Fixedness Parameters
K        = 50
ALPHA    = 0.6
LOG_BASE = 2
USE_LIN  = False

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
