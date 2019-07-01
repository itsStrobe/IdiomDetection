#!/bin/bash

# ============================================================ #

# Experiments 1
cd Experiments_1
. venv/bin/activate

# King and Cook Experiments - SVM - Embeddings + CForm
python3 Experiment.py

deactivate
cd ..

# ============================================================ #

# Experiments 2
cd Experiments_2
. venv/bin/activate

# SVM - CForm + Fazly's Fixedness Metrics
python3 Experiment_2_1.py

# King and Cook Experiments - SVM - Embeddings + CForm + Fazly's Fixedness Metrics
python3 Experiment_2_2.py

# Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics
python3 Experiment_2_3.py

# Unsupervised - Cosine Similarity
python3 Experiment_2_4.py

# Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness
python3 Experiment_2_5.py
python3 Experiment_2_5_b.py

deactivate
cd ..

# ============================================================ #

# Experiments 3
cd Experiments_3
. venv/bin/activate

# Unsupervised - Cosine Similarity [Candidates]
python3 Experiment_3_1.py

# Unsupervised - New Metrics -> Cosine Similarity and Overall Fixedness [Candidates]
python3 Experiment_3_2.py
python3 Experiment_3_2_b.py

deactivate
cd ..

# ============================================================ #
