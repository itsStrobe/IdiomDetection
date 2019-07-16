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
python36 ExtractVNICsInstances.py

echo "Extracting Potential VNICs Sentences from Corpora"
python36 ExtractCandidateSentences.py --NORM_FIX --USE_LIN

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
python36 GenerateWordEmbeddings.py

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
python GenerateWordEmbeddings.py

deactivate
cd ..

# ============================================================ #

# Skip-Thoughts
cd ./SkipThoughts

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python venv && source venv/bin/activate
pip install -r requirements.txt
################################

echo "Generating Skip-Thoughts Embeddings"
python GenerateWordEmbeddings.py

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
python36 GenerateWordEmbeddings.py

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
python36 Experiment_2_2.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_2/Clean/"

echo "Experient 2-2 : King and Cook Experiments - SVM - Embeddings Lemmatized + CForm"
python36 Experiment_2_2.py -VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_2/CForm_Lemm/" \
    --USE_CFORM

echo "Experient 2-2 : King and Cook Experiments - SVM - Embeddings Lemmatized + CForm + Fazly's Fixedness Metrics"
python36 Experiment_2_2.py -VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_2/CForm_Fix_Lemm/" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX

# Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics
echo "Experient 2-3 : Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics"
python36 Experiment_2_3.py

# Unsupervised - Cosine Similarity
echo "Experient 2-4 : Unsupervised - Cosine Similarity"
python36 Experiment_2_4.py

# Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness
echo "Experient 2-5 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness"
python36 Experiment_2_5.py
python36 Experiment_2_5_b.py

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
echo "Experient 3-1 : Unsupervised - Cosine Similarity [Candidates]"
python36 Experiment_3_1.py --VECTORS_FILE "embeddings_cand.csv" --VECTORS_FILE_VNC "embeddings_VNC_cand.csv" --RESULTS_DIR "./results/Experiment_3_1/Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_clean.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_clean.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_clean.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_clean.model"

echo "Experient 3-1 : Unsupervised - Cosine Similarity [Candidates] - Lemmatized"
python36 Experiment_3_1.py --VECTORS_FILE "embeddings_lemm_cand.csv" --VECTORS_FILE_VNC "embeddings_VNC_lemm_cand.csv" --RESULTS_DIR "./results/Experiment_3_1/Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm.model"

# Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates]
echo "Experient 3-2 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates]"
python36 Experiment_3_2.py --VECTORS_FILE "embeddings_cand.csv" --VECTORS_FILE_VNC "embeddings_VNC_cand.csv" --RESULTS_DIR "./results/Experiment_3_1/Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_clean.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_clean.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_clean.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_clean.model"

echo "Experient 3-2 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - Lemmatized"
python36 Experiment_3_2.py --VECTORS_FILE "embeddings_lemm_cand.csv" --VECTORS_FILE_VNC "embeddings_VNC_lemm_cand.csv" --RESULTS_DIR "./results/Experiment_3_1/Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm.model"

deactivate
cd ..

# ============================================================ #
# ============================================================ #
