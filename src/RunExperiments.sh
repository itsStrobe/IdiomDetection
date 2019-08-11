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

# King and Cook Experiments - SVM - Embeddings + CForm + Fazly's Fixedness Metrics
echo "Experiment 1-1 : King and Cook Experiments - SVM - Embeddings"
python3 Experiment_1_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_1_1/Clean/"

echo "Experiment 1-1 : King and Cook Experiments - SVM - Embeddings + CForm"
python3 Experiment_1_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_1_1/CForm/" \
    --USE_CFORM

echo "Experiment 1-1 : King and Cook Experiments - SVM - Embeddings + CForm + Fazly's Fixedness Metrics"
python3 Experiment_1_1.py -VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_1_1/CForm_Fix/" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX

echo "Experiment 1-1 : King and Cook Experiments - SVM - Embeddings Lemmatized"
python3 Experiment_1_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_1_1/Lemm/"

echo "Experiment 1-1 : King and Cook Experiments - SVM - Embeddings Lemmatized + CForm"
python3 Experiment_1_1.py -VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_1_1/CForm_Lemm/" \
    --USE_CFORM

echo "Experiment 1-1 : King and Cook Experiments - SVM - Embeddings Lemmatized + CForm + Fazly's Fixedness Metrics"
python3 Experiment_1_1.py -VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_1_1/CForm_Fix_Lemm/" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX

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

# Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics
echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings - 2 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k2/" --EXP_EXT "_k2" \
    --N_CLUSTERS 2 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm - 2 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k2/" --EXP_EXT "_k2_cForm" \
    --USE_CFORM    \
    --N_CLUSTERS 2 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics - 2 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k2/" --EXP_EXT "_k2_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 2 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings - 4 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k4/" --EXP_EXT "_k4" \
    --N_CLUSTERS 4 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm - 4 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k4/" --EXP_EXT "_k4_cForm" \
    --USE_CFORM    \
    --N_CLUSTERS 4 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics - 4 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k4/" --EXP_EXT "_k4_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 4 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings - 6 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k6/" --EXP_EXT "_k6" \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm - 6 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k6/" --EXP_EXT "_k6_cForm" \
    --USE_CFORM    \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics - 6 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k6/" --EXP_EXT "_k6_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings - 8 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k8/" --EXP_EXT "_k8" \
    --N_CLUSTERS 8 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm - 8 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k8/" --EXP_EXT "_k8_cForm" \
    --USE_CFORM    \
    --N_CLUSTERS 8 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics - 8 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k8/" --EXP_EXT "_k8_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 8 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings - 10 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k10/" --EXP_EXT "_k10" \
    --N_CLUSTERS 10 \
    --RND_STATE 42  \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm - 10 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k10/" --EXP_EXT "_k10_cForm" \
    --USE_CFORM \
    --N_CLUSTERS 10 \
    --RND_STATE 42  \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics - 10 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Clean_k10/" --EXP_EXT "_k10_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 10 \
    --RND_STATE 42  \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings - 2 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k2/" --EXP_EXT "_lemm_k2" \
    --N_CLUSTERS 2 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm - 2 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k2/" --EXP_EXT "_lemm_k2_cForm" \
    --USE_CFORM    \
    --N_CLUSTERS 2 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm + Fazly's Fixedness Metrics - 2 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k2/" --EXP_EXT "_lemm_k2_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 2 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings - 4 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k4/" --EXP_EXT "_lemm_k4" \
    --N_CLUSTERS 4 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm - 4 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k4/" --EXP_EXT "_lemm_k4_cForm" \
    --USE_CFORM \
    --N_CLUSTERS 4 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm + Fazly's Fixedness Metrics - 4 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k4/" --EXP_EXT "_lemm_k4_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 4 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings - 6 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k6/" --EXP_EXT "_lemm_k6" \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm - 6 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k6/" --EXP_EXT "_lemm_k6_cForm" \
    --USE_CFORM \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm + Fazly's Fixedness Metrics - 6 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k6/" --EXP_EXT "_lemm_k6_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings - 8 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k8/" --EXP_EXT "_lemm_k8" \
    --N_CLUSTERS 8 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm - 8 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k8/" --EXP_EXT "_lemm_k8_cForm" \
    --USE_CFORM \
    --N_CLUSTERS 8 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm + Fazly's Fixedness Metrics - 8 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k8/" --EXP_EXT "_lemm_k8_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 8 \
    --RND_STATE 42 \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings - 10 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k10/" --EXP_EXT "_lemm_k10" \
    --N_CLUSTERS 10 \
    --RND_STATE 42  \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm - 10 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k10/" --EXP_EXT "_lemm_k10_cForm" \
    --USE_CFORM \
    --N_CLUSTERS 10 \
    --RND_STATE 42  \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Lemmatized Embeddings + CForm + Fazly's Fixedness Metrics - 10 Clusters"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/Lemm_k10/" --EXP_EXT "_lemm_k10_cForm_fixMet" \
    --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX \
    --N_CLUSTERS 10 \
    --RND_STATE 42  \
    --SAVE_PLT

# ========== PMI========== #
# Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics
echo "Experiment 2-2 : Clustering Proposal - k-Means - Embeddings - 6 Clusters"
python36 Experiment_2_2.py --VECTORS_FILE "embeddings_cand_PMI.csv" --RESULTS_DIR "./results/Experiment_2_2/PMI/Clean_k2/" --EXP_EXT "_k2" \
    --SEED_VECTORS_FILE "embeddings.csv" \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLT

# ========== LEX========== #
# Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics
echo "Experiment 2-2 : Clustering Proposal - k-Means - Embeddings - 6 Clusters"
python36 Experiment_2_2.py --VECTORS_FILE "embeddings_cand_LEX.csv" --RESULTS_DIR "./results/Experiment_2_2/LEX/Clean_k2/" --EXP_EXT "_k2" \
    --SEED_VECTORS_FILE "embeddings.csv" \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLT

# ========== SYN========== #
# Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics
echo "Experiment 2-2 : Clustering Proposal - k-Means - Embeddings - 2 Clusters"
python36 Experiment_2_2.py --VECTORS_FILE "embeddings_cand_SYN.csv" --RESULTS_DIR "./results/Experiment_2_2/SYN/Clean_k2/" --EXP_EXT "_k2" \
    --SEED_VECTORS_FILE "embeddings.csv" \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLT

# ========== OVA========== #
# Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics
echo "Experiment 2-2 : Clustering Proposal - k-Means - Embeddings - 6 Clusters"
python36 Experiment_2_2.py --VECTORS_FILE "embeddings_cand_OVA.csv" --RESULTS_DIR "./results/Experiment_2_2/OVA/Clean_k2/" --EXP_EXT "_k2" \
    --SEED_VECTORS_FILE "embeddings.csv" \
    --N_CLUSTERS 6 \
    --RND_STATE 42 \
    --SAVE_PLT

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

# Unsupervised - Cosine Similarity
echo "Experiment 3-1 : Unsupervised - Cosine Similarity - Embeddings"
python3 Experiment_2_4.py --VECTORS_FILE "embeddings.csv" --VECTORS_FILE_VNC "embeddings_VNC.csv"  --RESULTS_DIR "./results/Experiment_3_1/Clean/" \

echo "Experiment 3-1 : Unsupervised - Cosine Similarity - Embeddings"
python3 Experiment_2_4.py --VECTORS_FILE "embeddings_lemm.csv" --VECTORS_FILE_VNC "embeddings_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_1/Lemm/" \

echo "Experiment 3-2 : Unsupervised - New Metrics - Embeddings -> Cosine Similarity and Overall Fixedness"
python3 Experiment_3_2.py --VECTORS_FILE "embeddings.csv" --VECTORS_FILE_VNC "embeddings_VNC.csv"  --RESULTS_DIR "./results/Experiment_3_2/Clean/"


echo "Experiment 3-2 : Unsupervised - New Metrics - Lemmatized Embeddings -> Cosine Similarity and Overall Fixedness"
python3 Experiment_3_2.py --VECTORS_FILE "embeddings_lemm.csv" --VECTORS_FILE_VNC "embeddings_VNC_lemm.csv"  --RESULTS_DIR "./results/Experiment_3_2/Lemm/"

# Unsupervised - Cosine Similarity [Candidates]
echo "Experiment 3-3 : Unsupervised - Cosine Similarity [Candidates] - PMI Candidates"
python3 Experiment_3_3.py --VECTORS_FILE "embeddings_cand_PMI.csv" --VECTORS_FILE_VNC "embeddings_cand_PMI_VNC.csv" --RESULTS_DIR "./results/Experiment_3_3/PMI_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_PMI.csv" \
    --COS_DIST_T 0.6

echo "Experiment 3-3 : Unsupervised - Cosine Similarity [Candidates] - PMI Candidates - Lemmatized"
python3 Experiment_3_3.py --VECTORS_FILE "embeddings_cand_PMI_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_PMI_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_3/PMI_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_PMI.csv" \
    --COS_DIST_T 0.6

echo "Experiment 3-3 : Unsupervised - Cosine Similarity [Candidates] - Lexical Fixedness Candidates"
python3 Experiment_3_3.py --VECTORS_FILE "embeddings_cand_LEX.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC.csv" --RESULTS_DIR "./results/Experiment_3_3/LEX_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_LEX.csv" \
    --COS_DIST_T 0.6

echo "Experiment 3-3 : Unsupervised - Cosine Similarity [Candidates] - Lexical Fixedness Candidates - Lemmatized"
python3 Experiment_3_3.py --VECTORS_FILE "embeddings_cand_LEX_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_3/LEX_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_LEX.csv" \
    --COS_DIST_T 0.6

echo "Experiment 3-3 : Unsupervised - Cosine Similarity [Candidates] - Syntactical Fixedness Candidates"
python3 Experiment_3_3.py --VECTORS_FILE "embeddings_cand_SYN.csv" --VECTORS_FILE_VNC "embeddings_cand_SYN_VNC.csv" --RESULTS_DIR "./results/Experiment_3_3/SYN_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_SYN.csv" \
    --COS_DIST_T 0.6

echo "Experiment 3-3 : Unsupervised - Cosine Similarity [Candidates] - Syntactical Fixedness Candidates - Lemmatized"
python3 Experiment_3_3.py --VECTORS_FILE "embeddings_cand_SYN_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_SYN_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_3/SYN_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_SYN.csv" \
    --COS_DIST_T 0.6

echo "Experiment 3-3 : Unsupervised - Cosine Similarity [Candidates] - Overall Fixedness Candidates"
python3 Experiment_3_3.py --VECTORS_FILE "embeddings_cand_OVA.csv" --VECTORS_FILE_VNC "embeddings_cand_OVA_VNC.csv" --RESULTS_DIR "./results/Experiment_3_3/OVA_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_OVA.csv" \
    --COS_DIST_T 0.6

echo "Experiment 3-3 : Unsupervised - Cosine Similarity [Candidates] - Overall Fixedness Candidates - Lemmatized"
python3 Experiment_3_3.py --VECTORS_FILE "embeddings_cand_OVA_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_OVA_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_3/OVA_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_OVA.csv" \
    --COS_DIST_T 0.6

# Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates]
echo "Experiment 3-4 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - PMI Candidates"
python3 Experiment_3_4.py --VECTORS_FILE "embeddings_cand_PMI.csv" --VECTORS_FILE_VNC "embeddings_cand_PMI_VNC.csv" --RESULTS_DIR "./results/Experiment_3_4/PMI_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_PMI.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experiment 3-4 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - PMI Candidates - Lemmatized"
python3 Experiment_3_4.py --VECTORS_FILE "embeddings_cand_PMI_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_PMI_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_4/PMI_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_PMI.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experiment 3-4 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - Lexical Fixedness Candidates"
python3 Experiment_3_4.py --VECTORS_FILE "embeddings_cand_LEX.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC.csv" --RESULTS_DIR "./results/Experiment_3_4/LEX_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_LEX.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experiment 3-4 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - Lexical Fixedness Candidates - Lemmatized"
python3 Experiment_3_4.py --VECTORS_FILE "embeddings_cand_LEX_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_4/LEX_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_LEX.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experiment 3-4 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - Syntactical Fixedness Candidates"
python3 Experiment_3_4.py --VECTORS_FILE "embeddings_cand_LEX.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC.csv" --RESULTS_DIR "./results/Experiment_3_4/LEX_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_SYN.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experiment 3-4 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - Syntactical Fixedness Candidates - Lemmatized"
python3 Experiment_3_4.py --VECTORS_FILE "embeddings_cand_LEX_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_4/LEX_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_lemm_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_lemm_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_lemm_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_SYN.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experiment 3-4 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - Overall Fixedness Candidates"
python3 Experiment_3_4.py --VECTORS_FILE "embeddings_cand_LEX.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC.csv" --RESULTS_DIR "./results/Experiment_3_4/LEX_Clean/" \
    --SVM_W2V   "../SVM_Models/W2V_cForms.model"   \
    --SVM_SCBOW "../SVM_Models/SCBOW_cForms.model" \
    --SVM_SKIP  "../SVM_Models/SKIP_cForms.model"  \
    --SVM_ELMO  "../SVM_Models/ELMO_cForms.model"  \
    --USE_CFORM \
    --CFORM_DIR "../targets/CForms_cand_OVA.csv" \
    --UNM_MET_T 0.4 --BETA 0.6

echo "Experiment 3-4 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates] - Overall Fixedness Candidates - Lemmatized"
python3 Experiment_3_4.py --VECTORS_FILE "embeddings_cand_LEX_lemm.csv" --VECTORS_FILE_VNC "embeddings_cand_LEX_VNC_lemm.csv" --RESULTS_DIR "./results/Experiment_3_4/LEX_Lemm/"\
    --SVM_W2V   "../SVM_Models/W2V_lemm_cForms.model"   \
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
