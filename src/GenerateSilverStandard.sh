#!/bin/bash

echo "GenerateSilverStandard.sh"

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python3 venv && source venv/bin/activate
pip3 install -r requirements.txt
################################

echo "Silver Standard for PMI Candidates"
# Get Silver Standard for PMI Candidates - Word2Vec - CForms
python3 VNICsCandidatesSilverStandard.py --DATASET_DIR "./targets/VNC-Tokens_candidates_PMI" --BEST_EMBD_DIR "./Word2Vec/embeddings_cand_PMI.csv" \
    --CFORM_DIR "./targets/CForms_cand_PMI.csv" \
    --BEST_SVM  "./SVM_Models/W2V_cForms.model" \
    --USE_CFORM

echo "Silver Standard for LEX Candidates"
# Get Silver Standard for LEX Candidates - Word2Vec - CForms
python3 VNICsCandidatesSilverStandard.py --DATASET_DIR "./targets/VNC-Tokens_candidates_LEX" --BEST_EMBD_DIR "./Word2Vec/embeddings_cand_LEX.csv" \
    --CFORM_DIR "./targets/CForms_cand_LEX.csv" \
    --BEST_SVM  "./SVM_Models/W2V_cForms.model" \
    --USE_CFORM

echo "Silver Standard for SYN Candidates"
# Get Silver Standard for SYN Candidates - Word2Vec - CForms
python3 VNICsCandidatesSilverStandard.py --DATASET_DIR "./targets/VNC-Tokens_candidates_SYN" --BEST_EMBD_DIR "./Word2Vec/embeddings_cand_SYN.csv" \
    --CFORM_DIR "./targets/CForms_cand_SYN.csv" \
    --BEST_SVM  "./SVM_Models/W2V_cForms.model" \
    --USE_CFORM 

echo "Silver Standard for OVA Candidates"
# Get Silver Standard for OVA Candidates - Word2Vec - CForms
python3 VNICsCandidatesSilverStandard.py --DATASET_DIR "./targets/VNC-Tokens_candidates_OVA" --BEST_EMBD_DIR "./Word2Vec/embeddings_cand_OVA.csv" \
    --CFORM_DIR "./targets/CForms_cand_OVA.csv" \
    --BEST_SVM  "./SVM_Models/W2V_cForms.model" \
    --USE_CFORM 

deactivate
