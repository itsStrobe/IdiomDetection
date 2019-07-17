import numpy as np
import pandas as pd
import progressbar

import VNCPatternCounts
from Util import CorpusExtraction

# ------------- ARGS ------------- #
parser = argparse.ArgumentParser()

parser.add_argument("--VNIC_FILE"     , "--vnic_file"         , type=str, help="Location of the CSV File Containing the Top Ranked VNICs Candidates.")
parser.add_argument("--VNIC_LOC_FILE" , "--vnic_dataset_file" , type=str, help="Location of the Output File Containing the VNICs Instances in VNC-Dataset Format.")

parser.add_argument("--TOP_N"        , "--top_n_candidates"    , type=int, help="Top N VNIC Candidates to be Used for Experiments.")
parser.add_argument("--MAX_WINDOW"   , "--maximum_window"      , type=int, help="Maximum Window Size for Extracting Candidate VNCIs from Corpora.")
parser.add_argument("--MAX_SENT_LEN" , "--maximum_sent_length" , type=int, help="Maximum Sentence Length to be considered for addition to the dataset.")

args = parser.parse_args()
# ------------- ARGS ------------- #

VNIC_FILE     = "./VNICs/PotentialVNICs_PMI.csv"
VNIC_LOC_FILE = "./targets/VNC-Tokens_candidates"

# Other Parameters
CORPORA_PRE   = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K"]
LEMMAS_SUF    = "_lemmas"
POSTAGS_SUF   = "_posTags"
LOC_TOKEN     = "/"
VNC_TOKEN     = "_"

# VNICs Instances Parameters
TOP_N        = 20
MAX_WINDOW   = 7
MAX_SENT_LEN = 80

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
                    # Only add instance if sentence length is smaller than a threshold - ELMo Overflow.
                    if(len(corpora_txt[corpus[s_n]]) < MAX_SENT_LEN):
                        vnics_token[it] = pat[0] + vnc_tok + pat[1]
                        vnics_loc[it]   = corp_loc
                        vnics_senId[it] = s_n
                    it += 1
                    if(it >= inst_n):
                        prog.update(it)
                        return

            prog.update(it)

def main():
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
    vnics = vnics[vnics.VNC != 'generic_vnic_instance_placeholder'] # Some instances occur in sentences without a proper ID, so they can't be added to the dataset.
    vnics.sort_values(['VNC'], inplace=True)

    vnics.to_csv(path_or_buf=VNIC_LOC_FILE, sep=' ', header=False, index=False)

if __name__ == '__main__':

    if(args.VNIC_FILE):
        VNIC_FILE = args.VNIC_FILE
    if(args.VNIC_LOC_FILE):
        VNIC_LOC_FILE = args.VNIC_LOC_FILE

    if(args.TOP_N):
        TOP_N = args.TOP_N
    if(args.MAX_WINDOW):
        MAX_WINDOW = args.MAX_WINDOW
    if(args.MAX_SENT_LEN):
        MAX_SENT_LEN = MAX_SENT_LEN

    main()
