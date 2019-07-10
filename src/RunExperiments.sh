#!/bin/bash
# Run all experiments

echo "RunExperiments.sh"

# ============================================================ #

# Experiments 1
cd Experiments_1
. venv/bin/activate

echo "Experiments 1"

# King and Cook Experiments - SVM - Embeddings + CForm
echo "Experient 1-1 : King and Cook Experiments - SVM - Embeddings + CForm"
# python3 Experiment.py

deactivate
cd ..

# ============================================================ #

# Experiments 2
cd Experiments_2
. venv/bin/activate

echo "Experiments 2"

# SVM - CForm + Fazly's Fixedness Metrics
echo "Experient 2-1 : SVM - CForm + Fazly's Fixedness Metrics"
python3 Experiment_2_1.py

# King and Cook Experiments - SVM - Embeddings + CForm + Fazly's Fixedness Metrics
echo "Experient 2-2 : King and Cook Experiments - SVM - Embeddings + CForm"
python3 Experiment_2_2.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_2/CForm/" --USE_CFORM
echo "Experient 2-2 : King and Cook Experiments - SVM - Embeddings + CForm + Fazly's Fixedness Metrics"
python3 Experiment_2_2.py -VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_2/CForm_Fix/" --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX
echo "Experient 2-2 : King and Cook Experiments - SVM - Embeddings Lemmatized + CForm"
python3 Experiment_2_2.py -VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_2/CForm_Lemm/" --USE_CFORM
echo "Experient 2-2 : King and Cook Experiments - SVM - Embeddings Lemmatized + CForm + Fazly's Fixedness Metrics"
python3 Experiment_2_2.py -VECTORS_FILE "embeddings_lemm.csv" --RESULTS_DIR "./results/Experiment_2_2/CForm_Fix_Lemm/" --USE_CFORM --USE_SYN_FIX --USE_LEX_FIX --USE_OVA_FIX

# Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics
echo "Experient 2-3 : Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics"
python3 Experiment_2_3.py

# Unsupervised - Cosine Similarity
echo "Experient 2-4 : Unsupervised - Cosine Similarity"
python3 Experiment_2_4.py

# Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness
echo "Experient 2-5 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness"
python3 Experiment_2_5.py
python3 Experiment_2_5_b.py

deactivate
cd ..

# ============================================================ #

# Experiments 3
cd Experiments_3
. venv/bin/activate

echo "Experiments 3"

# Unsupervised - Cosine Similarity [Candidates]
echo "Experient 3-1 : Unsupervised - Cosine Similarity [Candidates]"
python3 Experiment_3_1.py

# Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates]
echo "Experient 3-2 : Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates]"
python3 Experiment_3_2.py

deactivate
cd ..

# ============================================================ #
