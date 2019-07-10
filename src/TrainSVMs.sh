#!/bin/bash

echo "Training Embeddings + CForms + Fixedness"
# Train SVMs - Unlemmatized + CForms + Fixedness
python3 TrainSVMs.py --VECTORS_FILE "embeddings.csv" --MODELS_DIR "./SVM_Models/" --MODEL_EXT "_cForms_Fix" --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX --C_W2V 0.1 --C_SKIP 1 --C_SCBOW 1 --C_ELMo 1

echo "Training Lemmatized Embeddings + CForms + Fixedness"
# Train SVMs - Lemmatized + CForms + Fixedness
python3 TrainSVMs.py --VECTORS_FILE "embeddings_lemm.csv" --MODELS_DIR "./SVM_Models/" --MODEL_EXT "_lemm_cForms_Fix" --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX --C_W2V 0.1 --C_SKIP 1 --C_SCBOW 1 --C_ELMo 1
