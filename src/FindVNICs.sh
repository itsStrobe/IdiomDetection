#!/bin/bash
# Extract the Different Patterns Inside the Processed Corpus

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python3 venv && source venv/bin/activate
pip3 install -r requirements.txt
python3 -c "import nltk; nltk.download('wordnet')"
################################

echo "FindVNICs.sh"

echo "Extracting Patterns from Corpora"
python3 ExtractPatternsInCorpora.py

# Extract the Potential VNICs based on PMI

echo "Extracting Potential VNICs from Corpora"
python3 ExtractPotentialVNICs.py

echo "Extracting Potential VNICs Instances from Corpora"
python3 ExtractVNICsInstances.py --VNIC_FILE "./VNICs/PotentialVNICs_PMI.csv" --VNIC_LOC_FILE "./targets/VNC-Tokens_candidates_PMI" \
    --TOP_N 20 --MAX_WINDOW 7 --MAX_SENT_LEN 80

python3 ExtractVNICsInstances.py --VNIC_FILE "./VNICs/PotentialVNICs_LEX.csv" --VNIC_LOC_FILE "./targets/VNC-Tokens_candidates_LEX" \
    --TOP_N 20 --MAX_WINDOW 7 --MAX_SENT_LEN 80

python3 ExtractVNICsInstances.py --VNIC_FILE "./VNICs/PotentialVNICs_SYN.csv" --VNIC_LOC_FILE "./targets/VNC-Tokens_candidates_SYN" \
    --TOP_N 20 --MAX_WINDOW 7 --MAX_SENT_LEN 80

python3 ExtractVNICsInstances.py --VNIC_FILE "./VNICs/PotentialVNICs_OVA.csv" --VNIC_LOC_FILE "./targets/VNC-Tokens_candidates_OVA" \
    --TOP_N 20 --MAX_WINDOW 7 --MAX_SENT_LEN 80


echo "Extracting Potential VNICs Sentences from Corpora"
python3 ExtractCandidateSentences.py --TARG_CD_DIR "./targets/VNC-Tokens_candidates_PMI" \
    --SENT_CD_DIR         "./targets/Extracted_Sentences_cand_PMI.txt" \
    --SENT_LEMM_CD_DIR    "./targets/Extracted_Sentences_cand_PMI_lemm.txt" \
    --SENTVNC_CD_DIR      "./targets/Extracted_Sentences_cand_PMI_VNC.txt" \
    --SENTVNC_LEMM_CD_DIR "./targets/Extracted_Sentences_cand_PMI_VNC_lemm.txt" \
    --CFORM_CD_DIR   "./targets/CForms_cand_PMI.csv" \
    --SYN_FIX_CD_DIR "./targets/SynFix_cand_PMI.csv" \
    --LEX_FIX_CD_DIR "./targets/LexFix_cand_PMI.csv" \
    --OVA_FIX_CD_DIR "./targets/OvaFix_cand_PMI.csv" \
    --K 50 --ALPHA 0.6 --LOG_BASE 2 --MAX_WINDOW 7

python3 ExtractCandidateSentences.py --TARG_CD_DIR "./targets/VNC-Tokens_candidates_LEX" \
    --SENT_CD_DIR         "./targets/Extracted_Sentences_cand_LEX.txt" \
    --SENT_LEMM_CD_DIR    "./targets/Extracted_Sentences_cand_LEX_lemm.txt" \
    --SENTVNC_CD_DIR      "./targets/Extracted_Sentences_cand_LEX_VNC.txt" \
    --SENTVNC_LEMM_CD_DIR "./targets/Extracted_Sentences_cand_LEX_VNC_lemm.txt" \
    --CFORM_CD_DIR   "./targets/CForms_cand_LEX.csv" \
    --SYN_FIX_CD_DIR "./targets/SynFix_cand_LEX.csv" \
    --LEX_FIX_CD_DIR "./targets/LexFix_cand_LEX.csv" \
    --OVA_FIX_CD_DIR "./targets/OvaFix_cand_LEX.csv" \
    --K 50 --ALPHA 0.6 --LOG_BASE 2 --MAX_WINDOW 7

python3 ExtractCandidateSentences.py --TARG_CD_DIR "./targets/VNC-Tokens_candidates_SYN" \
    --SENT_CD_DIR         "./targets/Extracted_Sentences_cand_SYN.txt" \
    --SENT_LEMM_CD_DIR    "./targets/Extracted_Sentences_cand_SYN_lemm.txt" \
    --SENTVNC_CD_DIR      "./targets/Extracted_Sentences_cand_SYN_VNC.txt" \
    --SENTVNC_LEMM_CD_DIR "./targets/Extracted_Sentences_cand_SYN_VNC_lemm.txt" \
    --CFORM_CD_DIR   "./targets/CForms_cand_SYN.csv" \
    --SYN_FIX_CD_DIR "./targets/SynFix_cand_SYN.csv" \
    --LEX_FIX_CD_DIR "./targets/LexFix_cand_SYN.csv" \
    --OVA_FIX_CD_DIR "./targets/OvaFix_cand_SYN.csv" \
    --K 50 --ALPHA 0.6 --LOG_BASE 2 --MAX_WINDOW 7

python3 ExtractCandidateSentences.py --TARG_CD_DIR "./targets/VNC-Tokens_candidates_OVA" \
    --SENT_CD_DIR         "./targets/Extracted_Sentences_cand_OVA.txt" \
    --SENT_LEMM_CD_DIR    "./targets/Extracted_Sentences_cand_OVA_lemm.txt" \
    --SENTVNC_CD_DIR      "./targets/Extracted_Sentences_cand_OVA_VNC.txt" \
    --SENTVNC_LEMM_CD_DIR "./targets/Extracted_Sentences_cand_OVA_VNC_lemm.txt" \
    --CFORM_CD_DIR   "./targets/CForms_cand_OVA.csv" \
    --SYN_FIX_CD_DIR "./targets/SynFix_cand_OVA.csv" \
    --LEX_FIX_CD_DIR "./targets/LexFix_cand_OVA.csv" \
    --OVA_FIX_CD_DIR "./targets/OvaFix_cand_OVA.csv" \
    --K 50 --ALPHA 0.6 --LOG_BASE 2 --MAX_WINDOW 7

deactivate
