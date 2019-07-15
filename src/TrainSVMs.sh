#!/bin/bash

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python3 venv && source venv/bin/activate
pip3 install -r requirements.txt
################################

echo "Training Embeddings"
# Train SVMs - Unlemmatized + CForms + Fixedness
python3 TrainSVMs.py --VECTORS_FILE "embeddings.csv" --MODELS_DIR "./SVM_Models/" --MODEL_EXT "_clean" \
    --C_W2V 0.1 --C_SKIP 1 --C_SCBOW 1 --C_ELMo 1

echo "Training Embeddings + CForms"
# Train SVMs - Unlemmatized + CForms + Fixedness
python3 TrainSVMs.py --VECTORS_FILE "embeddings.csv" --MODELS_DIR "./SVM_Models/" --MODEL_EXT "_cForms" \
    --USE_CFORM \
    --C_W2V 0.1 --C_SKIP 1 --C_SCBOW 1 --C_ELMo 1

echo "Training Embeddings + CForms + Fixedness"
# Train SVMs - Unlemmatized + CForms + Fixedness
python3 TrainSVMs.py --VECTORS_FILE "embeddings.csv" --MODELS_DIR "./SVM_Models/" --MODEL_EXT "_cForms_Fix" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --C_W2V 0.1 --C_SKIP 1 --C_SCBOW 1 --C_ELMo 1

echo "Training Embeddings Lemmatized"
# Train SVMs - Lemmatized + CForms + Fixedness
python3 TrainSVMs.py --VECTORS_FILE "embeddings_lemm.csv" --MODELS_DIR "./SVM_Models/" --MODEL_EXT "_lemm" \
    --C_W2V 0.1 --C_SKIP 1 --C_SCBOW 1 --C_ELMo 1

echo "Training Embeddings Lemmatized + CForms"
# Train SVMs - Unlemmatized + CForms + Fixedness
python3 TrainSVMs.py --VECTORS_FILE "embeddings_lemm.csv" --MODELS_DIR "./SVM_Models/" --MODEL_EXT "_lemm_cForms" \
    --USE_CFORM \
    --C_W2V 0.1 --C_SKIP 1 --C_SCBOW 1 --C_ELMo 1

echo "Training Embeddings Lemmatized + CForms + Fixedness"
# Train SVMs - Lemmatized + CForms + Fixedness
python3 TrainSVMs.py --VECTORS_FILE "embeddings_lemm.csv" --MODELS_DIR "./SVM_Models/" --MODEL_EXT "_lemm_cForms_Fix" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --C_W2V 0.1 --C_SKIP 1 --C_SCBOW 1 --C_ELMo 1

deactivate
