#!/bin/bash
# Train Models with Corpora and Generate Embeddings for VNC-Dataset and VNC-Candidates

echo "GenerateEmbeddings.sh"

# ============================================================ #

# Word2Vec
cd ./Word2Vec

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python3 venv && source venv/bin/activate
pip3 install -r requirements.txt
################################\

echo "Generating Word2Vec Embeddings"
python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_VNC.txt" \
                                  --EMBD_DIR    "./embeddings.csv" \
                                  --EMBDVNC_DIR "./embeddings_VNC.csv" \
                                  --MODEL_DIR   "./models/W2V_ver1.model"

python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_VNC_lemm.csv" \
                                  --MODEL_DIR   "./models/W2V_ver1_lemm.model"

python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_PMI.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_PMI_VNC.csv" \
                                  --MODEL_DIR   "./models/W2V_ver1.model"

python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_PMI_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_PMI_VNC_lemm.csv" \
                                  --MODEL_DIR   "./models/W2V_ver1_lemm.model"

python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_LEX.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_LEX_VNC.csv" \
                                  --MODEL_DIR   "./models/W2V_ver1.model"

python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_LEX_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_LEX_VNC_lemm.csv" \
                                  --MODEL_DIR   "./models/W2V_ver1_lemm.model"

python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_SYN.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_SYN_VNC.csv" \
                                  --MODEL_DIR   "./models/W2V_ver1.model"

python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_SYN_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_SYN_VNC_lemm.csv" \
                                  --MODEL_DIR   "./models/W2V_ver1_lemm.model"

python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_OVA_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_OVA.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_OVA_VNC.csv" \
                                  --MODEL_DIR   "./models/W2V_ver1.model"

python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA_lemm.txt" \
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
virtualenv -p python3 venv && source venv/bin/activate
pip3 install -r requirements.txt
################################

echo "Generating ELMo Embeddings"
python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_VNC.txt" \
                                  --EMBD_DIR    "./embeddings.csv" \
                                  --EMBDVNC_DIR "./embeddings_VNC.csv" \
                                  --BATCH_SIZE  300

python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_VNC_lemm.csv" \
                                  --BATCH_SIZE  300

python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_PMI.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_PMI_VNC.csv" \
                                  --BATCH_SIZE  300

python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_PMI_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_PMI_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_PMI_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_PMI_VNC_lemm.csv" \
                                  --BATCH_SIZE  300

python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_LEX.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_LEX_VNC.csv" \
                                  --BATCH_SIZE  300

python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_LEX_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_LEX_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_LEX_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_LEX_VNC_lemm.csv" \
                                  --BATCH_SIZE  300

python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_SYN.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_SYN_VNC.csv" \
                                  --BATCH_SIZE  300

python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_SYN_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_SYN_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_SYN_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_SYN_VNC_lemm.csv" \
                                  --BATCH_SIZE  300

python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_OVA_VNC.txt" \
                                  --EMBD_DIR    "./embeddings_cand_OVA.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_OVA_VNC.csv" \
                                  --BATCH_SIZE  300

python3 GenerateWordEmbeddings.py --SENT_DIR    "../targets/Extracted_Sentences_cand_OVA_lemm.txt" \
                                  --SENTVNC_DIR "../targets/Extracted_Sentences_cand_OVA_VNC_lemm.txt" \
                                  --EMBD_DIR    "./embeddings_cand_OVA_lemm.csv" \
                                  --EMBDVNC_DIR "./embeddings_cand_OVA_VNC_lemm.csv" \
                                  --BATCH_SIZE  300

deactivate
cd ..
