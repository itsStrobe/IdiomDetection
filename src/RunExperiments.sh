#!/bin/bash
# Run all experiments

echo "RunExperiments.sh"

# ============================================================ #
# ============================================================ #

# Experiments 1
cd Experiments_1

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python3 venv && source venv/bin/activate
pip3 install -r requirements.txt
################################

echo "Experiments 1"

# King and Cook Experiments - SVM - Embeddings + CForm
echo "Experient 1-1 : King and Cook Experiments - SVM - Embeddings + CForm"
python3 Experiment.py

deactivate
cd ..

# ============================================================ #
# ============================================================ #

# Experiments 2
cd Experiments_2

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python3 venv && source venv/bin/activate
pip3 install -r requirements.txt
################################

echo "Experiments 2"

# SVM - CForm + Fazly's Fixedness Metrics
echo "Experient 2-1 : SVM - CForm + Fazly's Fixedness Metrics"
python3 Experiment_2_1.py

# King and Cook Experiments - SVM - Embeddings + CForm + Fazly's Fixedness Metrics
echo "Experient 2-2 : King and Cook Experiments - SVM - Embeddings"
python3 Experiment_2_2.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_2/Clean/"

echo "Experient 2-2 : King and Cook Experiments - SVM - Embeddings + CForm"
python3 Experiment_2_2.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_2/CForm/" \
    --USE_CFORM

echo "Experient 2-2 : King and Cook Experiments - SVM - Embeddings + CForm + Fazly's Fixedness Metrics"
python3 Experiment_2_2.py -VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_2/CForm_Fix/" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX

echo "Experient 2-2 : King and Cook Experiments - SVM - Embeddings Lemmatized"
python3 Experiment_2_2.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_2/Lemm/"

echo "Experient 2-2 : King and Cook Experiments - SVM - Embeddings Lemmatized + CForm"
python3 Experiment_2_2.py -VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_2/CForm_Lemm/" \
    --USE_CFORM

echo "Experient 2-2 : King and Cook Experiments - SVM - Embeddings Lemmatized + CForm + Fazly's Fixedness Metrics"
python3 Experiment_2_2.py -VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_2/CForm_Fix_Lemm/" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX

# Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics
echo "Experient 2-3 : Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics"
python3 Experiment_2_3.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_3/Clean/" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --SAVE_PLOT

echo "Experient 2-3 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm + Fazly's Fixedness Metrics"
python3 Experiment_2_3.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_3/Lemm/" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --SAVE_PLOT

# Unsupervised - Cosine Similarity
echo "Experient 2-4 : Unsupervised - Cosine Similarity - Embeddings"
python3 Experiment_2_4.py --VECTORS_FILE "embeddings.csv" --VECTORS_FILE_VNC "embeddings_VNC.csv"  --RESULTS_DIR "./results/Experiment_2_4/Clean/" \
    --COS_DIST_T 0.6

echo "Experient 2-4 : Unsupervised - Cosine Similarity - Embeddings"
python3 Experiment_2_4.py --VECTORS_FILE "embeddings_lemm.csv" --VECTORS_FILE_VNC "embeddings_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_2_4/Lemm/" \
    --COS_DIST_T 0.6

# Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness
echo "Experient 2-5 : Unsupervised - New Metrics - Embeddings -> Cosine Similarity and Overall Fixedness"
python3 Experiment_2_5.py --VECTORS_FILE "embeddings.csv" --VECTORS_FILE_VNC "embeddings_VNC.csv"  --RESULTS_DIR "./results/Experiment_2_5/Clean/" \
    --UNM_MET_T 0.7 --BETA 0.6

echo "Experient 2-5 : Unsupervised - New Metrics - Lemmatized Embeddings -> Cosine Similarity and Overall Fixedness"
python3 Experiment_2_5.py --VECTORS_FILE "embeddings_lemm.csv" --VECTORS_FILE_VNC "embeddings_VNC_lemm.csv"  --RESULTS_DIR "./results/Experiment_2_5/Lemm/" \
    --UNM_MET_T 0.7 --BETA 0.6

echo "Experient 2-5_b : Unsupervised - New Metrics - Embeddings -> Cosine Similarity and Overall Fixedness"
python3 Experiment_2_5_b.py --VECTORS_FILE "embeddings.csv" --VECTORS_FILE_VNC "embeddings_VNC.csv"  --RESULTS_DIR "./results/Experiment_2_5_b/Clean/"


echo "Experient 2-5_b : Unsupervised - New Metrics - Lemmatized Embeddings -> Cosine Similarity and Overall Fixedness"
python3 Experiment_2_5_b.py --VECTORS_FILE "embeddings_lemm.csv" --VECTORS_FILE_VNC "embeddings_VNC_lemm.csv"  --RESULTS_DIR "./results/Experiment_2_5_b/Lemm/"

deactivate
cd ..

# ============================================================ #
# ============================================================ #

# Experiments 3
cd Experiments_3

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python3 venv && source venv/bin/activate
pip3 install -r requirements.txt
################################

echo "Experiments 3"

# Unsupervised - Cosine Similarity [Candidates]
echo "Experient 3-1 : Unsupervised - Cosine Similarity [Candidates] - PMI Candidates"
python3 Experiment_3_1.py --VECTORS_FILE "embeddings_cand_PMI.csv" --VECTORS_FILE_VNC "embeddings_cand_PMI_VNC.csv" --RESULTS_DIR "./results/Experiment_3_1/PMI_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_PMI.csv" \
    --COS_DIST_T 0.6

echo "Experient 3-1 : Unsupervised - Cosine Similarity [Candidates] - PMI Candidates - Lemmatized"
python3 Experiment_3_1.py --VECTORS_FILE "embeddings_cand_PMI_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_PMI_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_1/PMI_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms_.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms_.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms_.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms_.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_PMI.csv" \
    --COS_DIST_T 0.6

echo "Experient 3-1 : Unsupervised - Cosine Similarity [Candidates] - Lexical Fixedness Candidates"
python3 Experiment_3_1.py --VECTORS_FILE "embeddings_cand_LEX.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC.csv" --RESULTS_DIR "./results/Experiment_3_1/LEX_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_LEX.csv" \
    --COS_DIST_T 0.6

echo "Experient 3-1 : Unsupervised - Cosine Similarity [Candidates] - Lexical Fixedness Candidates - Lemmatized"
python3 Experiment_3_1.py --VECTORS_FILE "embeddings_cand_LEX_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_1/LEX_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms_.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms_.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms_.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms_.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_LEX.csv" \
    --COS_DIST_T 0.6

echo "Experient 3-1 : Unsupervised - Cosine Similarity [Candidates] - Syntactical Fixedness Candidates"
python3 Experiment_3_1.py --VECTORS_FILE "embeddings_cand_SYN.csv" --VECTORS_FILE_VNC "embeddings_cand_SYN_VNC.csv" --RESULTS_DIR "./results/Experiment_3_1/SYN_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_SYN.csv" \
    --COS_DIST_T 0.6

echo "Experient 3-1 : Unsupervised - Cosine Similarity [Candidates] - Syntactical Fixedness Candidates - Lemmatized"
python3 Experiment_3_1.py --VECTORS_FILE "embeddings_cand_SYN_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_SYN_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_1/SYN_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms_.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms_.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms_.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms_.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_SYN.csv" \
    --COS_DIST_T 0.6

echo "Experient 3-1 : Unsupervised - Cosine Similarity [Candidates] - Overall Fixedness Candidates"
python3 Experiment_3_1.py --VECTORS_FILE "embeddings_cand_OVA.csv" --VECTORS_FILE_VNC "embeddings_cand_OVA_VNC.csv" --RESULTS_DIR "./results/Experiment_3_1/OVA_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_OVA.csv" \
    --COS_DIST_T 0.6

echo "Experient 3-1 : Unsupervised - Cosine Similarity [Candidates] - Overall Fixedness Candidates - Lemmatized"
python3 Experiment_3_1.py --VECTORS_FILE "embeddings_cand_OVA_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_OVA_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_1/OVA_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms_.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms_.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms_.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms_.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_OVA.csv" \
    --COS_DIST_T 0.6

# Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates]
echo "Experient 3-2 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - PMI Candidates"
python3 Experiment_3_2.py --VECTORS_FILE "embeddings_cand_PMI.csv" --VECTORS_FILE_VNC "embeddings_cand_PMI_VNC.csv" --RESULTS_DIR "./results/Experiment_3_2/PMI_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_PMI.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experient 3-2 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - PMI Candidates - Lemmatized"
python3 Experiment_3_2.py --VECTORS_FILE "embeddings_cand_PMI_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_PMI_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_2/PMI_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms_.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms_.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms_.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms_.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_PMI.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experient 3-2 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - Lexical Fixedness Candidates"
python3 Experiment_3_2.py --VECTORS_FILE "embeddings_cand_LEX.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC.csv" --RESULTS_DIR "./results/Experiment_3_2/LEX_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_LEX.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experient 3-2 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - Lexical Fixedness Candidates - Lemmatized"
python3 Experiment_3_2.py --VECTORS_FILE "embeddings_cand_LEX_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_2/LEX_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms_.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms_.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms_.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms_.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_LEX.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experient 3-2 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - Syntactical Fixedness Candidates"
python3 Experiment_3_2.py --VECTORS_FILE "embeddings_cand_LEX.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC.csv" --RESULTS_DIR "./results/Experiment_3_2/LEX_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_SYN.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experient 3-2 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - Syntactical Fixedness Candidates - Lemmatized"
python3 Experiment_3_2.py --VECTORS_FILE "embeddings_cand_LEX_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_2/LEX_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms_.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms_.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms_.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms_.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_SYN.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experient 3-2 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - Overall Fixedness Candidates"
python3 Experiment_3_2.py --VECTORS_FILE "embeddings_cand_LEX.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC.csv" --RESULTS_DIR "./results/Experiment_3_2/LEX_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_OVA.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experient 3-2 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - Overall Fixedness Candidates - Lemmatized"
python3 Experiment_3_2.py --VECTORS_FILE "embeddings_cand_LEX_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_2/LEX_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms_.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms_.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms_.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms_.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_OVA.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

deactivate
cd ..

# ============================================================ #
# ============================================================ #
