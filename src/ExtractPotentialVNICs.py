
import os
import re
import sys
import pickle
import numpy as np
import pandas as pd

import progressbar

from SynLexFixedness import SynLexFixedness

PAT_MODEL = "./PatternCounts/PatternCounts_130619.pickle"
VNIC_DIR  = "./VNICs/PotentialVNICs.csv"
FREQ_T    = 200

# Initialize SynLexFixedness
pat_SynLexFix = SynLexFixedness(modelDir=PAT_MODEL)
vncs_n = len(pat_SynLexFix.model)

verb_np = np.full(vncs_n, 'generic_verb')
noun_np = np.full(vncs_n, 'generic_noun')
pmis_np = np.full(vncs_n, -1000.0)

# Iterate over all VNCs
it = 0
prog = progressbar.ProgressBar(max_value=vncs_n)
prog.start()
for vnc in pat_SynLexFix.model:
    verb_np[it] = vnc[0]
    noun_np[it] = vnc[1]
    if(pat_SynLexFix.model[vnc][0] >= FREQ_T): pmis_np[it] = pat_SynLexFix.PMI(vnc[0], vnc[1])
    it += 1

    prog.update(it)

vnc_pmis = pd.DataFrame(index=range(vncs_n), data={'Verb': verb_np, 'Noun': noun_np, 'PMI': pmis_np})
print(vnc_pmis.head)

vnc_pmis.sort_values(['PMI'], inplace=True, ascending=False)
print(vnc_pmis.head)

vnc_pmis.to_csv(path_or_buf=VNIC_DIR, sep=',')
