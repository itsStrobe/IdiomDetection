#!/bin/bash
# FILE: RunAll.sh 
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu.q
#$ -l gpu=1
source /usr/local/gpuallocation.sh

echo "RunAll_Ceres.sh"

#==========================================================#
#----------------------------------------------------------#
#__________________________________________________________#

# - VIRTUAL EVIRONMENT SETUP - #
# virtualenv -p python36 venv && source venv/bin/activate
# pip3 install -r requirements.txt
# python36 -c "import nltk; nltk.download('wordnet')"
# ################################
# 
# echo "PrepareData.sh"
# 
# echo "Cleaning BNC"
# python36 CleanBNC.py
# 
# echo "Extracting Sentences from Corpora"
# python36 ExtractCorpora.py
# 
# echo "Extracting Text and Tags from Corpora"
# python36 ExtractTextAndTags.py
# 
# echo "Extract Instances of VNC-Dataset from Corpora"
# python36 ExtractTargetSentences.py
# 
# deactivate

#==========================================================#
#----------------------------------------------------------#
#__________________________________________________________#

echo "TrainEmbeddings.sh"

# ============================================================ #

# Word2Vec
# cd ./Word2Vec
# 
# # - VIRTUAL EVIRONMENT SETUP - #
# virtualenv -p python36 venv && source venv/bin/activate
# pip3 install -r requirements.txt
# ################################
# 
# echo "Training Word2Vec"
# python36 TrainWordEmbeddings.py
# 
# cp -r ./models/ ../
# 
# deactivate
# cd ..

# ============================================================ #

# Siamese CBOW
# cd ./SiameseCBOW
# 
# # - VIRTUAL EVIRONMENT SETUP - #
# virtualenv -p python venv && source venv/bin/activate
# pip install -r requirements.txt
# ################################
# 
# echo "Training SiameseCBOW"
# python TrainWordEmbeddings.py
# 
# deactivate
# cd ..

# ============================================================ #

# Skip-Thoughts
# cd ./SkipThoughts
# 
# # - VIRTUAL EVIRONMENT SETUP - #
# virtualenv -p python venv && source venv/bin/activate
# pip install -r requirements.txt
# python36 -c "import nltk; nltk.download('punkt')"
# ################################
# 
# echo "Training Skip-Thoughts"
# python TrainWordEmbeddings.py
# 
# deactivate
# cd ..

# ============================================================ #

# ELMo
# cd ./ELMo
# 
# - VIRTUAL EVIRONMENT SETUP - #
# virtualenv -p python36 venv && source venv/bin/activate
# pip3 install -r requirements.txt
# ################################
# 
# echo "Training ELMo"
# python36 TrainWordEmbeddings.py
# 
# deactivate
# cd ..

#==========================================================#
#----------------------------------------------------------#
#__________________________________________________________#

# echo "FindVNICs.sh"
# 
# # - VIRTUAL EVIRONMENT SETUP - #
# virtualenv -p python36 venv && source venv/bin/activate
# pip3 install -r requirements.txt
# python36 -c "import nltk; nltk.download('wordnet')"
# ################################
#  
# echo "Extracting Patterns from Corpora"
# python36 ExtractPatternsInCorpora.py
# 
# # Extract the Potential VNICs based on PMI
# 
# echo "Extracting Potential VNICs from Corpora"
# python36 ExtractPotentialVNICs.py
# 
# echo "Extracting Potential VNICs Instances from Corpora"
# python36 ExtractVNICsInstances.py --VNIC_FILE "./VNICs/PotentialVNICs_PMI.csv" --VNIC_LOC_FILE "./targets/VNC-Tokens_candidates_PMI" \
#     --TOP_N 20 --MAX_WINDOW 7 --MAX_SENT_LEN 80
# 
# python36 ExtractVNICsInstances.py --VNIC_FILE "./VNICs/PotentialVNICs_LEX.csv" --VNIC_LOC_FILE "./targets/VNC-Tokens_candidates_LEX" \
#     --TOP_N 20 --MAX_WINDOW 7 --MAX_SENT_LEN 80
# 
# python36 ExtractVNICsInstances.py --VNIC_FILE "./VNICs/PotentialVNICs_SYN.csv" --VNIC_LOC_FILE "./targets/VNC-Tokens_candidates_SYN" \
#     --TOP_N 20 --MAX_WINDOW 7 --MAX_SENT_LEN 80
# 
# python36 ExtractVNICsInstances.py --VNIC_FILE "./VNICs/PotentialVNICs_OVA.csv" --VNIC_LOC_FILE "./targets/VNC-Tokens_candidates_OVA" \
#     --TOP_N 20 --MAX_WINDOW 7 --MAX_SENT_LEN 80
# 
# 
# echo "Extracting Potential VNICs Sentences from Corpora"
# python36 ExtractCandidateSentences.py --TARG_CD_DIR "./targets/VNC-Tokens_candidates_PMI" \
#     --SENT_CD_DIR         "./targets/Extracted_Sentences_cand_PMI.txt" \
#     --SENT_LEMM_CD_DIR    "./targets/Extracted_Sentences_cand_PMI_lemm.txt" \
#     --SENTVNC_CD_DIR      "./targets/Extracted_Sentences_cand_PMI_VNC.txt" \
#     --SENTVNC_LEMM_CD_DIR "./targets/Extracted_Sentences_cand_PMI_VNC_lemm.txt" \
#     --CFORM_CD_DIR   "./targets/CForms_cand_PMI.csv" \
#     --SYN_FIX_CD_DIR "./targets/SynFix_cand_PMI.csv" \
#     --LEX_FIX_CD_DIR "./targets/LexFix_cand_PMI.csv" \
#     --OVA_FIX_CD_DIR "./targets/OvaFix_cand_PMI.csv" \
#     --K 50 --ALPHA 0.6 --MAX_WINDOW 7
# 
# python36 ExtractCandidateSentences.py --TARG_CD_DIR "./targets/VNC-Tokens_candidates_LEX" \
#     --SENT_CD_DIR         "./targets/Extracted_Sentences_cand_LEX.txt" \
#     --SENT_LEMM_CD_DIR    "./targets/Extracted_Sentences_cand_LEX_lemm.txt" \
#     --SENTVNC_CD_DIR      "./targets/Extracted_Sentences_cand_LEX_VNC.txt" \
#     --SENTVNC_LEMM_CD_DIR "./targets/Extracted_Sentences_cand_LEX_VNC_lemm.txt" \
#     --CFORM_CD_DIR   "./targets/CForms_cand_LEX.csv" \
#     --SYN_FIX_CD_DIR "./targets/SynFix_cand_LEX.csv" \
#     --LEX_FIX_CD_DIR "./targets/LexFix_cand_LEX.csv" \
#     --OVA_FIX_CD_DIR "./targets/OvaFix_cand_LEX.csv" \
#     --K 50 --ALPHA 0.6 --MAX_WINDOW 7
# 
# python36 ExtractCandidateSentences.py --TARG_CD_DIR "./targets/VNC-Tokens_candidates_SYN" \
#     --SENT_CD_DIR         "./targets/Extracted_Sentences_cand_SYN.txt" \
#     --SENT_LEMM_CD_DIR    "./targets/Extracted_Sentences_cand_SYN_lemm.txt" \
#     --SENTVNC_CD_DIR      "./targets/Extracted_Sentences_cand_SYN_VNC.txt" \
#     --SENTVNC_LEMM_CD_DIR "./targets/Extracted_Sentences_cand_SYN_VNC_lemm.txt" \
#     --CFORM_CD_DIR   "./targets/CForms_cand_SYN.csv" \
#     --SYN_FIX_CD_DIR "./targets/SynFix_cand_SYN.csv" \
#     --LEX_FIX_CD_DIR "./targets/LexFix_cand_SYN.csv" \
#     --OVA_FIX_CD_DIR "./targets/OvaFix_cand_SYN.csv" \
#     --K 50 --ALPHA 0.6 --MAX_WINDOW 7
# 
# python36 ExtractCandidateSentences.py --TARG_CD_DIR "./targets/VNC-Tokens_candidates_OVA" \
#     --SENT_CD_DIR         "./targets/Extracted_Sentences_cand_OVA.txt" \
#     --SENT_LEMM_CD_DIR    "./targets/Extracted_Sentences_cand_OVA_lemm.txt" \
#     --SENTVNC_CD_DIR      "./targets/Extracted_Sentences_cand_OVA_VNC.txt" \
#     --SENTVNC_LEMM_CD_DIR "./targets/Extracted_Sentences_cand_OVA_VNC_lemm.txt" \
#     --CFORM_CD_DIR   "./targets/CForms_cand_OVA.csv" \
#     --SYN_FIX_CD_DIR "./targets/SynFix_cand_OVA.csv" \
#     --LEX_FIX_CD_DIR "./targets/LexFix_cand_OVA.csv" \
#     --OVA_FIX_CD_DIR "./targets/OvaFix_cand_OVA.csv" \
#     --K 50 --ALPHA 0.6 --MAX_WINDOW 7
# 
# deactivate
# 
# #==========================================================#
# #----------------------------------------------------------#
# #__________________________________________________________#

# echo "GenerateEmbeddings.sh"
# 
# # ============================================================ #
# 
# # Word2Vec
# cd ./Word2Vec
# 
# # - VIRTUAL EVIRONMENT SETUP - #
# virtualenv -p python36 venv && source venv/bin/activate
# pip3 install -r requirements.txt
# ################################\
# 
# echo "Generating Word2Vec Embeddings"
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings.csv" \
#                                   --EMBDVNC_DIR "./embeddings_VNC.csv" \
#                                   --MODEL_DIR   "./models/W2V_ver1.model"
# 
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_VNC_lemm.csv" \
#                                   --MODEL_DIR   "./models/W2V_ver1_lemm.model"
# 
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_PMI.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_PMI_VNC.csv" \
#                                   --MODEL_DIR   "./models/W2V_ver1.model"
# 
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_PMI_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_PMI_VNC_lemm.csv" \
#                                   --MODEL_DIR   "./models/W2V_ver1_lemm.model"
# 
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_LEX.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_LEX_VNC.csv" \
#                                   --MODEL_DIR   "./models/W2V_ver1.model"
# 
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_LEX_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_LEX_VNC_lemm.csv" \
#                                   --MODEL_DIR   "./models/W2V_ver1_lemm.model"
# 
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_SYN.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_SYN_VNC.csv" \
#                                   --MODEL_DIR   "./models/W2V_ver1.model"
# 
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_SYN_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_SYN_VNC_lemm.csv" \
#                                   --MODEL_DIR   "./models/W2V_ver1_lemm.model"
# 
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_OVA_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_OVA.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_OVA_VNC.csv" \
#                                   --MODEL_DIR   "./models/W2V_ver1.model"
# 
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_OVA_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_OVA_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_OVA_VNC_lemm.csv" \
#                                   --MODEL_DIR   "./models/W2V_ver1_lemm.model"
# 
# deactivate
# cd ..
# 
# # ============================================================ #
# 
# # Siamese CBOW
# cd ./SiameseCBOW
# 
# # - VIRTUAL EVIRONMENT SETUP - #
# virtualenv -p python venv && source venv/bin/activate
# pip install -r requirements.txt
# ################################
# 
# echo "Generating SiameseCBOW Embeddings"
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings.csv" \
#                                   --EMBDVNC_DIR "./embeddings_VNC.csv"
# 
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_VNC_lemm.csv"
# 
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_PMI.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_PMI_VNC.csv"
# 
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_PMI_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_PMI_VNC_lemm.csv"
# 
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_LEX.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_LEX_VNC.csv"
# 
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_LEX_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_LEX_VNC_lemm.csv"
# 
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_SYN.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_SYN_VNC.csv"
# 
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_SYN_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_SYN_VNC_lemm.csv"
# 
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_OVA_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_OVA.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_OVA_VNC.csv"
# 
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_OVA_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_OVA_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_OVA_VNC_lemm.csv"
# 
# deactivate
# cd ..
# 
# # ============================================================ #
# 
# # Skip-Thoughts
# cd ./SkipThoughts
# 
# # - VIRTUAL EVIRONMENT SETUP - #
# virtualenv -p python venv && source venv/bin/activate
# pip3 install -r requirements.txt
# ################################
# 
# echo "Generating Skip-Thoughts Embeddings"
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings.csv" \
#                                   --EMBDVNC_DIR "./embeddings_VNC.csv"
# 
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_VNC_lemm.csv"
# 
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_PMI.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_PMI_VNC.csv"
# 
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_PMI_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_PMI_VNC_lemm.csv"
# 
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_LEX.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_LEX_VNC.csv"
# 
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_LEX_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_LEX_VNC_lemm.csv"
# 
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_SYN.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_SYN_VNC.csv"
# 
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_SYN_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_SYN_VNC_lemm.csv"
# 
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_OVA_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_OVA.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_OVA_VNC.csv"
# 
# python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_OVA_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_OVA_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_OVA_VNC_lemm.csv"
# 
# deactivate
# cd ..
# 
# # ============================================================ #
# 
# # ELMo
# cd ./ELMo
# 
# # - VIRTUAL EVIRONMENT SETUP - #
# virtualenv -p python36 venv && source venv/bin/activate
# pip3 install -r requirements.txt
# ################################
# 
# echo "Generating ELMo Embeddings"
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings.csv" \
#                                   --EMBDVNC_DIR "./embeddings_VNC.csv" \
#                                   --BATCH_SIZE  300
# 
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_VNC_lemm.csv" \
#                                   --BATCH_SIZE  300
# 
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_PMI.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_PMI_VNC.csv" \
#                                   --BATCH_SIZE  300
# 
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_PMI_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_PMI_VNC_lemm.csv" \
#                                   --BATCH_SIZE  300
# 
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_LEX.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_LEX_VNC.csv" \
#                                   --BATCH_SIZE  300
# 
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_LEX_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_LEX_VNC_lemm.csv" \
#                                   --BATCH_SIZE  300
# 
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_SYN.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_SYN_VNC.csv" \
#                                   --BATCH_SIZE  300
# 
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_SYN_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_SYN_VNC_lemm.csv" \
#                                   --BATCH_SIZE  300
# 
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_OVA_VNC.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_OVA.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_OVA_VNC.csv" \
#                                   --BATCH_SIZE  300
# 
# python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA_lemm.txt" \
#                                   --SENTVNC_DIR "../targets/Extracted_Sentences_cand_OVA_VNC_lemm.txt" \
#                                   --EMBD_DIR    "./embeddings_cand_OVA_lemm.csv" \
#                                   --EMBDVNC_DIR "./embeddings_cand_OVA_VNC_lemm.csv" \
#                                   --BATCH_SIZE  300
# 
# deactivate
# cd ..
# 
# #==========================================================#
# #----------------------------------------------------------#
# #__________________________________________________________#

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python36 venv && source venv/bin/activate
pip3 install -r requirements.txt
################################

echo "Training Embeddings"
# Train SVMs - Unlemmatized + CForms + Fixedness
python36 TrainSVMs.py --VECTORS_FILE "embeddings.csv" --MODELS_DIR "./SVM_Models/" --MODEL_EXT "_clean" \
    --C_W2V 0.1 --C_SKIP 1 --C_SCBOW 1 --C_ELMo 1

echo "Training Embeddings + CForms"
# Train SVMs - Unlemmatized + CForms + Fixedness
python36 TrainSVMs.py --VECTORS_FILE "embeddings.csv" --MODELS_DIR "./SVM_Models/" --MODEL_EXT "_cForms" \
    --USE_CFORM \
    --C_W2V 0.1 --C_SKIP 1 --C_SCBOW 1 --C_ELMo 1

echo "Training Embeddings + CForms + Fixedness"
# Train SVMs - Unlemmatized + CForms + Fixedness
python36 TrainSVMs.py --VECTORS_FILE "embeddings.csv" --MODELS_DIR "./SVM_Models/" --MODEL_EXT "_cForms_Fix" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --C_W2V 0.1 --C_SKIP 1 --C_SCBOW 1 --C_ELMo 1

echo "Training Embeddings Lemmatized"
# Train SVMs - Lemmatized + CForms + Fixedness
python36 TrainSVMs.py --VECTORS_FILE "embeddings_lemm.csv" --MODELS_DIR "./SVM_Models/" --MODEL_EXT "_lemm" \
    --C_W2V 0.1 --C_SKIP 1 --C_SCBOW 1 --C_ELMo 1

echo "Training Embeddings Lemmatized + CForms"
# Train SVMs - Unlemmatized + CForms + Fixedness
python36 TrainSVMs.py --VECTORS_FILE "embeddings_lemm.csv" --MODELS_DIR "./SVM_Models/" --MODEL_EXT "_lemm_cForms" \
    --USE_CFORM \
    --C_W2V 0.1 --C_SKIP 1 --C_SCBOW 1 --C_ELMo 1

echo "Training Embeddings Lemmatized + CForms + Fixedness"
# Train SVMs - Lemmatized + CForms + Fixedness
python36 TrainSVMs.py --VECTORS_FILE "embeddings_lemm.csv" --MODELS_DIR "./SVM_Models/" --MODEL_EXT "_lemm_cForms_Fix" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --C_W2V 0.1 --C_SKIP 1 --C_SCBOW 1 --C_ELMo 1

deactivate


#==========================================================#
#----------------------------------------------------------#
#__________________________________________________________#

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python36 venv && source venv/bin/activate
pip3 install -r requirements.txt
################################

echo "Silver Standard for PMI Candidates"
# Get Silver Standard for PMI Candidates - Word2Vec - CForms
python36 VNICsCandidatesSilverStandard.py --DATASET_DIR "./targets/VNC-Tokens_candidates_PMI" --BEST_EMBD_DIR "./Word2Vec/embeddings_cand_PMI.csv" \
    --CFORM_DIR "./targets/CForms_cand_PMI.csv" \
    --BEST_SVM  "./SVM_Models/W2V_cForms.model" \
    --USE_CFORM

echo "Silver Standard for LEX Candidates"
# Get Silver Standard for LEX Candidates - Word2Vec - CForms
python36 VNICsCandidatesSilverStandard.py --DATASET_DIR "./targets/VNC-Tokens_candidates_LEX" --BEST_EMBD_DIR "./Word2Vec/embeddings_cand_LEX.csv" \
    --CFORM_DIR "./targets/CForms_cand_LEX.csv" \
    --BEST_SVM  "./SVM_Models/W2V_cForms.model" \
    --USE_CFORM

echo "Silver Standard for SYN Candidates"
# Get Silver Standard for SYN Candidates - Word2Vec - CForms
python36 VNICsCandidatesSilverStandard.py --DATASET_DIR "./targets/VNC-Tokens_candidates_SYN" --BEST_EMBD_DIR "./Word2Vec/embeddings_cand_SYN.csv" \
    --CFORM_DIR "./targets/CForms_cand_SYN.csv" \
    --BEST_SVM  "./SVM_Models/W2V_cForms.model" \
    --USE_CFORM 

echo "Silver Standard for OVA Candidates"
# Get Silver Standard for OVA Candidates - Word2Vec - CForms
python36 VNICsCandidatesSilverStandard.py --DATASET_DIR "./targets/VNC-Tokens_candidates_OVA" --BEST_EMBD_DIR "./Word2Vec/embeddings_cand_OVA.csv" \
    --CFORM_DIR "./targets/CForms_cand_OVA.csv" \
    --BEST_SVM  "./SVM_Models/W2V_cForms.model" \
    --USE_CFORM 

deactivate


#==========================================================#
#----------------------------------------------------------#
#__________________________________________________________#

echo "RunExperiments.sh"

# ============================================================ #
# ============================================================ #

# Experiments 1
cd Experiments_1

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python36 venv && source venv/bin/activate
pip3 install -r requirements.txt
################################

echo "Experiments 1"

# King and Cook Experiments - SVM - Embeddings + CForm + Fazly's Fixedness Metrics
echo "Experiment 1-1 : King and Cook Experiments - SVM - Embeddings"
python3 Experiment_1_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_1_1/Clean/"

echo "Experiment 1-1 : King and Cook Experiments - SVM - Embeddings + CForm"
python3 Experiment_1_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_1_1/CForm/" \
    --USE_CFORM

echo "Experiment 1-1 : King and Cook Experiments - SVM - Embeddings + CForm + Fazly's Fixedness Metrics"
python3 Experiment_1_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_1_1/CForm_Fix/" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX

echo "Experiment 1-1 : King and Cook Experiments - SVM - Embeddings Lemmatized"
python3 Experiment_1_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_1_1/Lemm/"

echo "Experiment 1-1 : King and Cook Experiments - SVM - Embeddings Lemmatized + CForm"
python3 Experiment_1_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_1_1/CForm_Lemm/" \
    --USE_CFORM

echo "Experiment 1-1 : King and Cook Experiments - SVM - Embeddings Lemmatized + CForm + Fazly's Fixedness Metrics"
python3 Experiment_1_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_1_1/CForm_Fix_Lemm/" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX

deactivate
cd ..

# ============================================================ #
# ============================================================ #

# Experiments 2
cd Experiments_2

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python36 venv && source venv/bin/activate
pip3 install -r requirements.txt
################################

echo "Experiments 2"

# Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics
echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings - 2 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k2/" --EXP_EXT "_k2" \
    --N_CLUSTERS 2 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm - 2 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k2/" --EXP_EXT "_k2_cForm" \
    --USE_CFORM    \
    --N_CLUSTERS 2 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics - 2 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k2/" --EXP_EXT "_k2_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 2 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings - 4 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k4/" --EXP_EXT "_k4" \
    --N_CLUSTERS 4 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm - 4 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k4/" --EXP_EXT "_k4_cForm" \
    --USE_CFORM    \
    --N_CLUSTERS 4 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics - 4 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k4/" --EXP_EXT "_k4_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 4 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings - 6 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k6/" --EXP_EXT "_k6" \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm - 6 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k6/" --EXP_EXT "_k6_cForm" \
    --USE_CFORM    \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics - 6 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k6/" --EXP_EXT "_k6_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings - 8 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k8/" --EXP_EXT "_k8" \
    --N_CLUSTERS 8 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm - 8 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k8/" --EXP_EXT "_k8_cForm" \
    --USE_CFORM    \
    --N_CLUSTERS 8 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics - 8 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k8/" --EXP_EXT "_k8_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 8 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings - 10 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k10/" --EXP_EXT "_k10" \
    --N_CLUSTERS 10 \
    --RND_STATE 42  \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm - 10 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k10/" --EXP_EXT "_k10_cForm" \
    --USE_CFORM \
    --N_CLUSTERS 10 \
    --RND_STATE 42  \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics - 10 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k10/" --EXP_EXT "_k10_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 10 \
    --RND_STATE 42  \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings - 2 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k2/" --EXP_EXT "_lemm_k2" \
    --N_CLUSTERS 2 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm - 2 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k2/" --EXP_EXT "_lemm_k2_cForm" \
    --USE_CFORM    \
    --N_CLUSTERS 2 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm + Fazly's Fixedness Metrics - 2 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k2/" --EXP_EXT "_lemm_k2_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 2 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings - 4 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k4/" --EXP_EXT "_lemm_k4" \
    --N_CLUSTERS 4 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm - 4 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k4/" --EXP_EXT "_lemm_k4_cForm" \
    --USE_CFORM \
    --N_CLUSTERS 4 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm + Fazly's Fixedness Metrics - 4 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k4/" --EXP_EXT "_lemm_k4_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 4 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings - 6 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k6/" --EXP_EXT "_lemm_k6" \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm - 6 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k6/" --EXP_EXT "_lemm_k6_cForm" \
    --USE_CFORM \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm + Fazly's Fixedness Metrics - 6 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k6/" --EXP_EXT "_lemm_k6_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings - 8 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k8/" --EXP_EXT "_lemm_k8" \
    --N_CLUSTERS 8 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm - 8 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k8/" --EXP_EXT "_lemm_k8_cForm" \
    --USE_CFORM \
    --N_CLUSTERS 8 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm + Fazly's Fixedness Metrics - 8 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k8/" --EXP_EXT "_lemm_k8_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 8 \
    --RND_STATE 42 \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings - 10 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k10/" --EXP_EXT "_lemm_k10" \
    --N_CLUSTERS 10 \
    --RND_STATE 42  \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm - 10 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k10/" --EXP_EXT "_lemm_k10_cForm" \
    --USE_CFORM \
    --N_CLUSTERS 10 \
    --RND_STATE 42  \
    --SAVE_PLOT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm + Fazly's Fixedness Metrics - 10 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k10/" --EXP_EXT "_lemm_k10_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 10 \
    --RND_STATE 42  \
    --SAVE_PLOT

# ========== PMI========== #
# Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics
echo "Experiment 2-2 : Clustering Proposal - k-Means - Embeddings - 6 Clusters"
python36 Experiment_2_2.py --VECTORS_FILE "embeddings_cand_PMI.csv" --RESULTS_DIR "./results/Experiment_2_2/PMI/Clean_k2/" --EXP_EXT "_k2" \
    --SEED_VECTORS_FILE "embeddings.csv" \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLOT

# ========== LEX========== #
# Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics
echo "Experiment 2-2 : Clustering Proposal - k-Means - Embeddings - 6 Clusters"
python36 Experiment_2_2.py --VECTORS_FILE "embeddings_cand_LEX.csv" --RESULTS_DIR "./results/Experiment_2_2/LEX/Clean_k2/" --EXP_EXT "_k2" \
    --SEED_VECTORS_FILE "embeddings.csv" \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLOT

# ========== SYN========== #
# Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics
echo "Experiment 2-2 : Clustering Proposal - k-Means - Embeddings - 2 Clusters"
python36 Experiment_2_2.py --VECTORS_FILE "embeddings_cand_SYN.csv" --RESULTS_DIR "./results/Experiment_2_2/SYN/Clean_k2/" --EXP_EXT "_k2" \
    --SEED_VECTORS_FILE "embeddings.csv" \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLOT

# ========== OVA========== #
# Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics
echo "Experiment 2-2 : Clustering Proposal - k-Means - Embeddings - 6 Clusters"
python36 Experiment_2_2.py --VECTORS_FILE "embeddings_cand_OVA.csv" --RESULTS_DIR "./results/Experiment_2_2/OVA/Clean_k2/" --EXP_EXT "_k2" \
    --SEED_VECTORS_FILE "embeddings.csv" \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLOT

deactivate
cd ..

# ============================================================ #
# ============================================================ #

# Experiments 3
cd Experiments_3

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python36 venv && source venv/bin/activate
pip3 install -r requirements.txt
################################

echo "Experiments 3"

# Unsupervised - Cosine Similarity
echo "Experiment 3-1 : Unsupervised - Cosine Similarity - Embeddings"
python3 Experiment_3_1.py --VECTORS_FILE "embeddings.csv" --VECTORS_FILE_VNC "embeddings_VNC.csv"  --RESULTS_DIR "./results/Experiment_3_1/Clean/" \

echo "Experiment 3-1 : Unsupervised - Cosine Similarity - Embeddings"
python3 Experiment_3_1.py --VECTORS_FILE "embeddings_lemm.csv" --VECTORS_FILE_VNC "embeddings_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_1/Lemm/" \

# Unsupervised - Cosine Similarity + CForm
echo "Experiment 3-2 : Unsupervised - New Metrics - Embeddings -> Cosine Similarity and CForm"
python3 Experiment_3_2.py --VECTORS_FILE "embeddings.csv" --VECTORS_FILE_VNC "embeddings_VNC.csv"  --RESULTS_DIR "./results/Experiment_3_2/Clean/"

echo "Experiment 3-2 : Unsupervised - New Metrics - Lemmatized Embeddings -> Cosine Similarity and CForm"
python3 Experiment_3_2.py --VECTORS_FILE "embeddings_lemm.csv" --VECTORS_FILE_VNC "embeddings_VNC_lemm.csv"  --RESULTS_DIR "./results/Experiment_3_2/Lemm/"

# Unsupervised - Cosine Similarity [Candidates]
echo "Experiment 3-3 : Unsupervised - Cosine Similarity [Candidates] - PMI Candidates"
python3 Experiment_3_3.py --VECTORS_FILE "embeddings_cand_PMI.csv" --VECTORS_FILE_VNC "embeddings_cand_PMI_VNC.csv" --RESULTS_DIR "./results/Experiment_3_3/PMI_Clean/" \
    --COS_DIST_T 0.6

echo "Experiment 3-3 : Unsupervised - Cosine Similarity [Candidates] - PMI Candidates - Lemmatized"
python3 Experiment_3_3.py --VECTORS_FILE "embeddings_cand_PMI_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_PMI_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_3/PMI_Lemm/"\
    --COS_DIST_T 0.6

echo "Experiment 3-3 : Unsupervised - Cosine Similarity [Candidates] - Lexical Fixedness Candidates"
python3 Experiment_3_3.py --VECTORS_FILE "embeddings_cand_LEX.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC.csv" --RESULTS_DIR "./results/Experiment_3_3/LEX_Clean/" \
    --COS_DIST_T 0.6

echo "Experiment 3-3 : Unsupervised - Cosine Similarity [Candidates] - Lexical Fixedness Candidates - Lemmatized"
python3 Experiment_3_3.py --VECTORS_FILE "embeddings_cand_LEX_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_3/LEX_Lemm/"\
    --COS_DIST_T 0.6

echo "Experiment 3-3 : Unsupervised - Cosine Similarity [Candidates] - Syntactical Fixedness Candidates"
python3 Experiment_3_3.py --VECTORS_FILE "embeddings_cand_SYN.csv" --VECTORS_FILE_VNC "embeddings_cand_SYN_VNC.csv" --RESULTS_DIR "./results/Experiment_3_3/SYN_Clean/" \
    --COS_DIST_T 0.6

echo "Experiment 3-3 : Unsupervised - Cosine Similarity [Candidates] - Syntactical Fixedness Candidates - Lemmatized"
python3 Experiment_3_3.py --VECTORS_FILE "embeddings_cand_SYN_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_SYN_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_3/SYN_Lemm/"\
    --COS_DIST_T 0.6

echo "Experiment 3-3 : Unsupervised - Cosine Similarity [Candidates] - Overall Fixedness Candidates"
python3 Experiment_3_3.py --VECTORS_FILE "embeddings_cand_OVA.csv" --VECTORS_FILE_VNC "embeddings_cand_OVA_VNC.csv" --RESULTS_DIR "./results/Experiment_3_3/OVA_Clean/" \
    --COS_DIST_T 0.6

echo "Experiment 3-3 : Unsupervised - Cosine Similarity [Candidates] - Overall Fixedness Candidates - Lemmatized"
python3 Experiment_3_3.py --VECTORS_FILE "embeddings_cand_OVA_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_OVA_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_3/OVA_Lemm/"\
    --COS_DIST_T 0.6

# Unsupervised - Cosine Similarity + CForm [Candidates]
echo "Experiment 3-4 : Unsupervised - New Metrics -> Cosine Similarity and CForm [Candidates] - PMI Candidates"
python3 Experiment_3_4.py --VECTORS_FILE "embeddings_cand_PMI.csv" --VECTORS_FILE_VNC "embeddings_cand_PMI_VNC.csv" --RESULTS_DIR "./results/Experiment_3_4/PMI_Clean/" \
    --CFORM_DIR "../targets/CForms_cand_PMI.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experiment 3-4 : Unsupervised - New Metrics -> Cosine Similarity and CForm [Candidates] - PMI Candidates - Lemmatized"
python3 Experiment_3_4.py --VECTORS_FILE "embeddings_cand_PMI_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_PMI_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_4/PMI_Lemm/"\
    --CFORM_DIR "../targets/CForms_cand_PMI.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experiment 3-4 : Unsupervised - New Metrics -> Cosine Similarity and CForm [Candidates] - Lexical Fixedness Candidates"
python3 Experiment_3_4.py --VECTORS_FILE "embeddings_cand_LEX.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC.csv" --RESULTS_DIR "./results/Experiment_3_4/LEX_Clean/" \
    --CFORM_DIR "../targets/CForms_cand_LEX.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experiment 3-4 : Unsupervised - New Metrics -> Cosine Similarity and CForm [Candidates] - Lexical Fixedness Candidates - Lemmatized"
python3 Experiment_3_4.py --VECTORS_FILE "embeddings_cand_LEX_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_4/LEX_Lemm/"\
    --CFORM_DIR "../targets/CForms_cand_LEX.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experiment 3-4 : Unsupervised - New Metrics -> Cosine Similarity and CForm [Candidates] - Syntactical Fixedness Candidates"
python3 Experiment_3_4.py --VECTORS_FILE "embeddings_cand_SYN.csv" --VECTORS_FILE_VNC "embeddings_cand_SYN_VNC.csv" --RESULTS_DIR "./results/Experiment_3_4/SYN_Clean/" \
    --CFORM_DIR "../targets/CForms_cand_SYN.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experiment 3-4 : Unsupervised - New Metrics -> Cosine Similarity and CForm [Candidates] - Syntactical Fixedness Candidates - Lemmatized"
python3 Experiment_3_4.py --VECTORS_FILE "embeddings_cand_SYN_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_SYN_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_4/SYN_Lemm/"\
    --CFORM_DIR "../targets/CForms_cand_SYN.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experiment 3-4 : Unsupervised - New Metrics -> Cosine Similarity and CForm [Candidates] - Overall Fixedness Candidates"
python3 Experiment_3_4.py --VECTORS_FILE "embeddings_cand_OVA.csv" --VECTORS_FILE_VNC "embeddings_cand_OVA_VNC.csv" --RESULTS_DIR "./results/Experiment_3_4/OVA_Clean/" \
    --CFORM_DIR "../targets/CForms_cand_OVA.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experiment 3-4 : Unsupervised - New Metrics -> Cosine Similarity and CForm [Candidates] - Overall Fixedness Candidates - Lemmatized"
python3 Experiment_3_4.py --VECTORS_FILE "embeddings_cand_OVA_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_OVA_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_4/OVA_Lemm/"\
    --CFORM_DIR "../targets/CForms_cand_OVA.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

deactivate
cd ..

# ============================================================ #
# ============================================================ #
