import numpy as np
import pandas as pd
import progressbar

import VNCPatternCounts
from Util import CorpusExtraction

CORPORA_PRE   = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K"]
VNIC_FILE     = "./VNICs/PotentialVNICs_PMI.csv"
VNIC_LOC_FILE = "./targets/VNC-Tokens_candidates"
LEMMAS_SUF    = "_lemmas"
POSTAGS_SUF   = "_posTags"
LOC_TOKEN     = "/"
VNC_TOKEN     = "_"

TOP_N      = 5
MAX_WINDOW = 7

def GetVNCsSet(verbs, nouns):
    vncs = []
    for verb, noun in zip(verbs, nouns):
        vncs.append((verb, noun))

    return vncs

def FindInstances(corpora_list, tok, loc, sId, inst_n, loc_tok=LOC_TOKEN, vnc_tok=VNC_TOKEN, lemmas_suf=LEMMAS_SUF, postags_suf=POSTAGS_SUF):
    it = 0
    prog = progressbar.ProgressBar(max_value=inst_n)
    prog.start()
    for corpora in corpora_list:
        corpora_txt = CorpusExtraction.LoadCorpora(corpora, suffix=lemmas_suf)
        corpora_pos = CorpusExtraction.LoadCorpora(corpora, suffix=postags_suf)

        for corpus in corpora_txt:
            corp_loc = corpus[:1] + loc_tok + corpus[:2] + loc_tok + corpus[:3]
            for s_n in corpora_txt[corpus]:
                pats = set([(pat[0], pat[1]) for pat in VNCPatternCounts.ExtractPatternsFromSentence(corpora_txt[corpus][s_n], corpora_pos[corpus][s_n], max_window=MAX_WINDOW) if (pat[0], pat[1]) in vnics_set])
                for pat in pats:
                    vnics_token[it] = pat[0] + vnc_tok + pat[1]
                    vnics_loc[it]   = corp_loc
                    vnics_senId[it] = s_n
                    it += 1
                    if(it >= inst_n):
                        prog.update(it)
                        return

            prog.update(it)


# Extract Top N candidate VNICs
vnics_pd = pd.read_csv(VNIC_FILE, header=0, index_col=False, nrows=TOP_N, usecols=['Verb', 'Noun', 'Freq'])
print(vnics_pd)

instances = 0
for freq in vnics_pd['Freq'].values:
    instances += int(freq)

vnics_set = set(GetVNCsSet(vnics_pd['Verb'].values, vnics_pd['Noun'].values))

vnics_class = np.full(instances, 'Q')
vnics_token = np.full(instances, 'generic_vnic_instance_placeholder')
vnics_loc   = np.full(instances, 'X/XX/XXX')
vnics_senId = np.full(instances, -1)

FindInstances(CORPORA_PRE, vnics_token, vnics_loc, vnics_senId, instances)

vnics = pd.DataFrame(index=range(instances), data={'Class': vnics_class, 'VNC': vnics_token, 'Corpus': vnics_loc, 'Sent ID': vnics_senId})
vnics.sort_values(['VNC'], inplace=True)

vnics.to_csv(path_or_buf=VNIC_LOC_FILE, sep=' ', header=False, index=False)

