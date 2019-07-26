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

echo "FindVNICs.sh"

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python36 venv && source venv/bin/activate
pip3 install -r requirements.txt
python36 -c "import nltk; nltk.download('wordnet')"
################################
 
echo "Extracting Patterns from Corpora"
python36 ExtractPatternsInCorpora.py

# Extract the Potential VNICs based on PMI

echo "Extracting Potential VNICs from Corpora"
python36 ExtractPotentialVNICs.py

echo "Extracting Potential VNICs Instances from Corpora"
python36 ExtractVNICsInstances.py --VNIC_FILE "./VNICs/PotentialVNICs_PMI.csv" --VNIC_LOC_FILE "./targets/VNC-Tokens_candidates_PMI" \
    --TOP_N 20 --MAX_WINDOW 7 --MAX_SENT_LEN 80

python36 ExtractVNICsInstances.py --VNIC_FILE "./VNICs/PotentialVNICs_LEX.csv" --VNIC_LOC_FILE "./targets/VNC-Tokens_candidates_LEX" \
    --TOP_N 20 --MAX_WINDOW 7 --MAX_SENT_LEN 80

python36 ExtractVNICsInstances.py --VNIC_FILE "./VNICs/PotentialVNICs_SYN.csv" --VNIC_LOC_FILE "./targets/VNC-Tokens_candidates_SYN" \
    --TOP_N 20 --MAX_WINDOW 7 --MAX_SENT_LEN 80

python36 ExtractVNICsInstances.py --VNIC_FILE "./VNICs/PotentialVNICs_OVA.csv" --VNIC_LOC_FILE "./targets/VNC-Tokens_candidates_OVA" \
    --TOP_N 20 --MAX_WINDOW 7 --MAX_SENT_LEN 80


echo "Extracting Potential VNICs Sentences from Corpora"
python36 ExtractCandidateSentences.py --TARG_CD_DIR "./targets/VNC-Tokens_candidates_PMI" \
    --SENT_CD_DIR         "./targets/Extracted_Sentences_cand_PMI.txt" \
    --SENT_LEMM_CD_DIR    "./targets/Extracted_Sentences_cand_PMI_lemm.txt" \
    --SENTVNC_CD_DIR      "./targets/Extracted_Sentences_cand_PMI_VNC.txt" \
    --SENTVNC_LEMM_CD_DIR "./targets/Extracted_Sentences_cand_PMI_VNC_lemm.txt" \
    --CFORM_CD_DIR   "./targets/CForms_cand_PMI.csv" \
    --SYN_FIX_CD_DIR "./targets/SynFix_cand_PMI.csv" \
    --LEX_FIX_CD_DIR "./targets/LexFix_cand_PMI.csv" \
    --OVA_FIX_CD_DIR "./targets/OvaFix_cand_PMI.csv" \
    --K 50 --ALPHA 0.6 --MAX_WINDOW 7

python36 ExtractCandidateSentences.py --TARG_CD_DIR "./targets/VNC-Tokens_candidates_LEX" \
    --SENT_CD_DIR         "./targets/Extracted_Sentences_cand_LEX.txt" \
    --SENT_LEMM_CD_DIR    "./targets/Extracted_Sentences_cand_LEX_lemm.txt" \
    --SENTVNC_CD_DIR      "./targets/Extracted_Sentences_cand_LEX_VNC.txt" \
    --SENTVNC_LEMM_CD_DIR "./targets/Extracted_Sentences_cand_LEX_VNC_lemm.txt" \
    --CFORM_CD_DIR   "./targets/CForms_cand_LEX.csv" \
    --SYN_FIX_CD_DIR "./targets/SynFix_cand_LEX.csv" \
    --LEX_FIX_CD_DIR "./targets/LexFix_cand_LEX.csv" \
    --OVA_FIX_CD_DIR "./targets/OvaFix_cand_LEX.csv" \
    --K 50 --ALPHA 0.6 --MAX_WINDOW 7

python36 ExtractCandidateSentences.py --TARG_CD_DIR "./targets/VNC-Tokens_candidates_SYN" \
    --SENT_CD_DIR         "./targets/Extracted_Sentences_cand_SYN.txt" \
    --SENT_LEMM_CD_DIR    "./targets/Extracted_Sentences_cand_SYN_lemm.txt" \
    --SENTVNC_CD_DIR      "./targets/Extracted_Sentences_cand_SYN_VNC.txt" \
    --SENTVNC_LEMM_CD_DIR "./targets/Extracted_Sentences_cand_SYN_VNC_lemm.txt" \
    --CFORM_CD_DIR   "./targets/CForms_cand_SYN.csv" \
    --SYN_FIX_CD_DIR "./targets/SynFix_cand_SYN.csv" \
    --LEX_FIX_CD_DIR "./targets/LexFix_cand_SYN.csv" \
    --OVA_FIX_CD_DIR "./targets/OvaFix_cand_SYN.csv" \
    --K 50 --ALPHA 0.6 --MAX_WINDOW 7

python36 ExtractCandidateSentences.py --TARG_CD_DIR "./targets/VNC-Tokens_candidates_OVA" \
    --SENT_CD_DIR         "./targets/Extracted_Sentences_cand_OVA.txt" \
    --SENT_LEMM_CD_DIR    "./targets/Extracted_Sentences_cand_OVA_lemm.txt" \
    --SENTVNC_CD_DIR      "./targets/Extracted_Sentences_cand_OVA_VNC.txt" \
    --SENTVNC_LEMM_CD_DIR "./targets/Extracted_Sentences_cand_OVA_VNC_lemm.txt" \
    --CFORM_CD_DIR   "./targets/CForms_cand_OVA.csv" \
    --SYN_FIX_CD_DIR "./targets/SynFix_cand_OVA.csv" \
    --LEX_FIX_CD_DIR "./targets/LexFix_cand_OVA.csv" \
    --OVA_FIX_CD_DIR "./targets/OvaFix_cand_OVA.csv" \
    --K 50 --ALPHA 0.6 --MAX_WINDOW 7

deactivate

#==========================================================#
#----------------------------------------------------------#
#__________________________________________________________#

echo "GenerateEmbeddings.sh"

# ============================================================ #

# Word2Vec
cd ./Word2Vec

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python36 venv && source venv/bin/activate
pip3 install -r requirements.txt
################################\

echo "Generating Word2Vec Embeddings"
python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_VNC.txt" \
                                  --EMBD_DIR    "./embeddings.csv" \
                                  --EMBDVNC_DIR "./embeddings_VNC.csv" \
                                  --MODEL_DIR   "./models/W2V_ver1.model"

python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_VNC_lemm.csv" \
                                  --MODEL_DIR   "./models/W2V_ver1_lemm.model"

python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_PMI.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_PMI_VNC.csv" \
                                  --MODEL_DIR   "./models/W2V_ver1.model"

python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_PMI_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_PMI_VNC_lemm.csv" \
                                  --MODEL_DIR   "./models/W2V_ver1_lemm.model"

python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_LEX.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_LEX_VNC.csv" \
                                  --MODEL_DIR   "./models/W2V_ver1.model"

python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_LEX_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_LEX_VNC_lemm.csv" \
                                  --MODEL_DIR   "./models/W2V_ver1_lemm.model"

python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_SYN.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_SYN_VNC.csv" \
                                  --MODEL_DIR   "./models/W2V_ver1.model"

python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_SYN_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_SYN_VNC_lemm.csv" \
                                  --MODEL_DIR   "./models/W2V_ver1_lemm.model"

python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_OVA_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_OVA.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_OVA_VNC.csv" \
                                  --MODEL_DIR   "./models/W2V_ver1.model"

python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_OVA_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_OVA_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_OVA_VNC_lemm.csv" \
                                  --MODEL_DIR   "./models/W2V_ver1_lemm.model"

deactivate
cd ..

# ============================================================ #

# Siamese CBOW
cd ./SiameseCBOW

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python venv && source venv/bin/activate
pip install -r requirements.txt
################################

echo "Generating SiameseCBOW Embeddings"
python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_VNC.txt" \
                                  --EMBD_DIR    "./embeddings.csv" \
                                  --EMBDVNC_DIR "./embeddings_VNC.csv"

python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_VNC_lemm.csv"

python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_PMI.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_PMI_VNC.csv"

python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_PMI_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_PMI_VNC_lemm.csv"

python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_LEX.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_LEX_VNC.csv"

python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_LEX_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_LEX_VNC_lemm.csv"

python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_SYN.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_SYN_VNC.csv"

python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_SYN_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_SYN_VNC_lemm.csv"

python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_OVA_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_OVA.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_OVA_VNC.csv"

python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_OVA_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_OVA_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_OVA_VNC_lemm.csv"

deactivate
cd ..

# ============================================================ #

# Skip-Thoughts
cd ./SkipThoughts

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python venv && source venv/bin/activate
pip3 install -r requirements.txt
################################

echo "Generating Skip-Thoughts Embeddings"
python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_VNC.txt" \
                                  --EMBD_DIR    "./embeddings.csv" \
                                  --EMBDVNC_DIR "./embeddings_VNC.csv"

python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_VNC_lemm.csv"

python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_PMI.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_PMI_VNC.csv"

python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_PMI_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_PMI_VNC_lemm.csv"

python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_LEX.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_LEX_VNC.csv"

python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_LEX_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_LEX_VNC_lemm.csv"

python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_SYN.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_SYN_VNC.csv"

python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_SYN_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_SYN_VNC_lemm.csv"

python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_OVA_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_OVA.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_OVA_VNC.csv"

python GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_OVA_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_OVA_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_OVA_VNC_lemm.csv"

deactivate
cd ..

# ============================================================ #

# ELMo
cd ./ELMo

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python36 venv && source venv/bin/activate
pip3 install -r requirements.txt
################################

echo "Generating ELMo Embeddings"
python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_VNC.txt" \
                                  --EMBD_DIR    "./embeddings.csv" \
                                  --EMBDVNC_DIR "./embeddings_VNC.csv" \
                                  --BATCH_SIZE  300

python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_VNC_lemm.csv" \
                                  --BATCH_SIZE  300

python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_PMI.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_PMI_VNC.csv" \
                                  --BATCH_SIZE  300

python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_PMI_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_PMI_VNC_lemm.csv" \
                                  --BATCH_SIZE  300

python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_LEX.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_LEX_VNC.csv" \
                                  --BATCH_SIZE  300

python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_LEX_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_LEX_VNC_lemm.csv" \
                                  --BATCH_SIZE  300

python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_SYN.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_SYN_VNC.csv" \
                                  --BATCH_SIZE  300

python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_SYN_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_SYN_VNC_lemm.csv" \
                                  --BATCH_SIZE  300

python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_OVA_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_OVA.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_OVA_VNC.csv" \
                                  --BATCH_SIZE  300

python36 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_OVA_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_OVA_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_OVA_VNC_lemm.csv" \
                                  --BATCH_SIZE  300

deactivate
cd ..

#==========================================================#
#----------------------------------------------------------#
#__________________________________________________________#

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

# King and Cook Experiments - SVM - Embeddings + CForm
echo "Experient 1-1 : King and Cook Experiments - SVM - Embeddings + CForm"
python36 Experiment.py

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

# SVM - CForm + Fazly's Fixedness Metrics
echo "Experient 2-1 : SVM - CForm + Fazly's Fixedness Metrics"
python36 Experiment_2_1.py

# King and Cook Experiments - SVM - Embeddings + CForm + Fazly's Fixedness Metrics
echo "Experient 2-2 : King and Cook Experiments - SVM - Embeddings"
python36 Experiment_2_2.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_2/Clean/"

echo "Experient 2-2 : King and Cook Experiments - SVM - Embeddings + CForm"
python36 Experiment_2_2.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_2/CForm/" \
    --USE_CFORM

echo "Experient 2-2 : King and Cook Experiments - SVM - Embeddings + CForm + Fazly's Fixedness Metrics"
python36 Experiment_2_2.py -VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_2/CForm_Fix/" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX

echo "Experient 2-2 : King and Cook Experiments - SVM - Embeddings Lemmatized"
python36 Experiment_2_2.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_2/Lemm/"

echo "Experient 2-2 : King and Cook Experiments - SVM - Embeddings Lemmatized + CForm"
python36 Experiment_2_2.py -VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_2/CForm_Lemm/" \
    --USE_CFORM

echo "Experient 2-2 : King and Cook Experiments - SVM - Embeddings Lemmatized + CForm + Fazly's Fixedness Metrics"
python36 Experiment_2_2.py -VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_2/CForm_Fix_Lemm/" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX

# Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics
echo "Experient 2-3 : Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics"
python36 Experiment_2_3.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_3/Clean/" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --SAVE_PLOT

echo "Experient 2-3 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm + Fazly's Fixedness Metrics"
python36 Experiment_2_3.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_3/Lemm/" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --SAVE_PLOT

# Unsupervised - Cosine Similarity
echo "Experient 2-4 : Unsupervised - Cosine Similarity - Embeddings"
python36 Experiment_2_4.py --VECTORS_FILE "embeddings.csv" --VECTORS_FILE_VNC "embeddings_VNC.csv"  --RESULTS_DIR "./results/Experiment_2_4/Clean/" \
    --COS_DIST_T 0.6

echo "Experient 2-4 : Unsupervised - Cosine Similarity - Embeddings"
python36 Experiment_2_4.py --VECTORS_FILE "embeddings_lemm.csv" --VECTORS_FILE_VNC "embeddings_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_2_4/Lemm/" \
    --COS_DIST_T 0.6

# Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness
echo "Experient 2-5 : Unsupervised - New Metrics - Embeddings -> Cosine Similarity and Overall Fixedness"
python36 Experiment_2_5.py --VECTORS_FILE "embeddings.csv" --VECTORS_FILE_VNC "embeddings_VNC.csv"  --RESULTS_DIR "./results/Experiment_2_5/Clean/" \
    --UNM_MET_T 0.7 --BETA 0.6

echo "Experient 2-5 : Unsupervised - New Metrics - Lemmatized Embeddings -> Cosine Similarity and Overall Fixedness"
python36 Experiment_2_5.py --VECTORS_FILE "embeddings_lemm.csv" --VECTORS_FILE_VNC "embeddings_VNC_lemm.csv"  --RESULTS_DIR "./results/Experiment_2_5/Lemm/" \
    --UNM_MET_T 0.7 --BETA 0.6

echo "Experient 2-5_b : Unsupervised - New Metrics - Embeddings -> Cosine Similarity and Overall Fixedness"
python36 Experiment_2_5_b.py --VECTORS_FILE "embeddings.csv" --VECTORS_FILE_VNC "embeddings_VNC.csv"  --RESULTS_DIR "./results/Experiment_2_5_b/Clean/"


echo "Experient 2-5_b : Unsupervised - New Metrics - Lemmatized Embeddings -> Cosine Similarity and Overall Fixedness"
python36 Experiment_2_5_b.py --VECTORS_FILE "embeddings_lemm.csv" --VECTORS_FILE_VNC "embeddings_VNC_lemm.csv"  --RESULTS_DIR "./results/Experiment_2_5_b/Lemm/"

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

# Unsupervised - Cosine Similarity [Candidates]
echo "Experient 3-1 : Unsupervised - Cosine Similarity [Candidates] - PMI Candidates"
python36 Experiment_3_1.py --OG_SENT_DIR "../targets/Extracted_Sentences_cand_PMI.txt" --VECTORS_FILE "embeddings_cand_PMI.csv" --VECTORS_FILE_VNC "embeddings_cand_PMI_VNC.csv" --RESULTS_DIR "./results/Experiment_3_1/PMI_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_PMI.csv" \
    --COS_DIST_T 0.6

echo "Experient 3-1 : Unsupervised - Cosine Similarity [Candidates] - PMI Candidates - Lemmatized"
python36 Experiment_3_1.py --OG_SENT_DIR "../targets/Extracted_Sentences_cand_PMI_lemm.txt" --VECTORS_FILE "embeddings_cand_PMI_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_PMI_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_1/PMI_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms_.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms_.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms_.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms_.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_PMI.csv" \
    --COS_DIST_T 0.6

echo "Experient 3-1 : Unsupervised - Cosine Similarity [Candidates] - Lexical Fixedness Candidates"
python36 Experiment_3_1.py --OG_SENT_DIR "../targets/Extracted_Sentences_cand_LEX.txt" --VECTORS_FILE "embeddings_cand_LEX.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC.csv" --RESULTS_DIR "./results/Experiment_3_1/LEX_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_LEX.csv" \
    --COS_DIST_T 0.6

echo "Experient 3-1 : Unsupervised - Cosine Similarity [Candidates] - Lexical Fixedness Candidates - Lemmatized"
python36 Experiment_3_1.py --OG_SENT_DIR "../targets/Extracted_Sentences_cand_LEX_lemm.txt" --VECTORS_FILE "embeddings_cand_LEX_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_1/LEX_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms_.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms_.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms_.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms_.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_LEX.csv" \
    --COS_DIST_T 0.6

echo "Experient 3-1 : Unsupervised - Cosine Similarity [Candidates] - Syntactical Fixedness Candidates"
python36 Experiment_3_1.py --OG_SENT_DIR "../targets/Extracted_Sentences_cand_SYN.txt" --VECTORS_FILE "embeddings_cand_SYN.csv" --VECTORS_FILE_VNC "embeddings_cand_SYN_VNC.csv" --RESULTS_DIR "./results/Experiment_3_1/SYN_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_SYN.csv" \
    --COS_DIST_T 0.6

echo "Experient 3-1 : Unsupervised - Cosine Similarity [Candidates] - Syntactical Fixedness Candidates - Lemmatized"
python36 Experiment_3_1.py --OG_SENT_DIR "../targets/Extracted_Sentences_cand_LEX_lemm.txt" --VECTORS_FILE "embeddings_cand_SYN_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_SYN_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_1/SYN_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms_.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms_.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms_.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms_.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_SYN.csv" \
    --COS_DIST_T 0.6

echo "Experient 3-1 : Unsupervised - Cosine Similarity [Candidates] - Overall Fixedness Candidates"
python36 Experiment_3_1.py --OG_SENT_DIR "../targets/Extracted_Sentences_cand_OVA.txt" --VECTORS_FILE "embeddings_cand_OVA.csv" --VECTORS_FILE_VNC "embeddings_cand_OVA_VNC.csv" --RESULTS_DIR "./results/Experiment_3_1/OVA_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_OVA.csv" \
    --COS_DIST_T 0.6

echo "Experient 3-1 : Unsupervised - Cosine Similarity [Candidates] - Overall Fixedness Candidates - Lemmatized"
python36 Experiment_3_1.py --OG_SENT_DIR "../targets/Extracted_Sentences_cand_OVA_lemm.txt" --VECTORS_FILE "embeddings_cand_OVA_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_OVA_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_1/OVA_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms_.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms_.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms_.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms_.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_OVA.csv" \
    --COS_DIST_T 0.6

# Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates]
echo "Experient 3-2 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - PMI Candidates"
python36 Experiment_3_2.py --OG_SENT_DIR "../targets/Extracted_Sentences_cand_PMI.txt" --VECTORS_FILE "embeddings_cand_PMI.csv" --VECTORS_FILE_VNC "embeddings_cand_PMI_VNC.csv" --RESULTS_DIR "./results/Experiment_3_2/PMI_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_PMI.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experient 3-2 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - PMI Candidates - Lemmatized"
python36 Experiment_3_2.py --OG_SENT_DIR "../targets/Extracted_Sentences_cand_PMI_lemm.txt" --VECTORS_FILE "embeddings_cand_PMI_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_PMI_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_2/PMI_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms_.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms_.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms_.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms_.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_PMI.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experient 3-2 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - Lexical Fixedness Candidates"
python36 Experiment_3_2.py --OG_SENT_DIR "../targets/Extracted_Sentences_cand_LEX.txt" --VECTORS_FILE "embeddings_cand_LEX.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC.csv" --RESULTS_DIR "./results/Experiment_3_2/LEX_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_LEX.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experient 3-2 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - Lexical Fixedness Candidates - Lemmatized"
python36 Experiment_3_2.py --OG_SENT_DIR "../targets/Extracted_Sentences_cand_LEX_lemm.txt" --VECTORS_FILE "embeddings_cand_LEX_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_2/LEX_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms_.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms_.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms_.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms_.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_LEX.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experient 3-2 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - Syntactical Fixedness Candidates"
python36 Experiment_3_2.py --OG_SENT_DIR "../targets/Extracted_Sentences_cand_SYN.txt" --VECTORS_FILE "embeddings_cand_SYN.csv" --VECTORS_FILE_VNC "embeddings_cand_SYN_VNC.csv" --RESULTS_DIR "./results/Experiment_3_2/SYN_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_SYN.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experient 3-2 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - Syntactical Fixedness Candidates - Lemmatized"
python36 Experiment_3_2.py --OG_SENT_DIR "../targets/Extracted_Sentences_cand_SYN_lemm.txt" --VECTORS_FILE "embeddings_cand_SYN_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_SYN_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_2/SYN_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_SYN.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experient 3-2 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - Overall Fixedness Candidates"
python36 Experiment_3_2.py --OG_SENT_DIR "../targets/Extracted_Sentences_cand_OVA.txt" --VECTORS_FILE "embeddings_cand_OVA.csv" --VECTORS_FILE_VNC "embeddings_cand_OVA_VNC.csv" --RESULTS_DIR "./results/Experiment_3_2/OVA_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_OVA.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experient 3-2 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - Overall Fixedness Candidates - Lemmatized"
python36 Experiment_3_2.py --OG_SENT_DIR "../targets/Extracted_Sentences_cand_OVA_lemm.txt" --VECTORS_FILE "embeddings_cand_OVA_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_OVA_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_2/OVA_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms_.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_OVA.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

deactivate
cd ..

# ============================================================ #
# ============================================================ #
