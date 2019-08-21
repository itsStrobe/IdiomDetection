#!/bin/bash
# FILE: Run_GetVNICCandidates.sh 
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu.q
#$ -l gpu=1
source /usr/local/gpuallocation.sh

echo "Run_kMeansELMoTest.sh"

# - VIRTUAL EVIRONMENT SETUP - #
cd Experiments_2

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python36 venv && source venv/bin/activate
pip3 install -r requirements.txt
################################

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics - 10 Clusters - Cluster 1"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/ELMo_Test_k10/" --EXP_EXT "_k10_ELMo_Test_1" \
    --N_CLUSTERS 10 \
    --RND_STATE 10  \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics - 10 Clusters - Cluster 2"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/ELMo_Test_k10/" --EXP_EXT "_k10_ELMo_Test_2" \
    --N_CLUSTERS 10 \
    --RND_STATE 20  \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics - 10 Clusters - Cluster 3"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/ELMo_Test_k10/" --EXP_EXT "_k10_ELMo_Test_3" \
    --N_CLUSTERS 10 \
    --RND_STATE 30  \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics - 10 Clusters - Cluster 4"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/ELMo_Test_k10/" --EXP_EXT "_k10_ELMo_Test_4" \
    --N_CLUSTERS 10 \
    --RND_STATE 40  \
    --SAVE_PLT

echo "Experiment 2-1 : Clustering Proposal - k-Means - Embeddings + CForm + Fazly's Fixedness Metrics - 10 Clusters - Cluster 5"
python36 Experiment_2_1.py --VECTORS_FILE "embeddings.csv" --RESULTS_DIR "./results/Experiment_2_1/VNC-Tokens/ELMo_Test_k10/" --EXP_EXT "_k10_ELMo_Test_5" \
    --N_CLUSTERS 10 \
    --RND_STATE 50  \
    --SAVE_PLT

deactivate
cd ..
