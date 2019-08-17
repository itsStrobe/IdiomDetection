#!/bin/bash
# FILE: Run_GetVNICCandidates.sh 
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu.q
#$ -l gpu=1
source /usr/local/gpuallocation.sh

echo "Run_GetVNICCandidates.sh"

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python36 venv && source venv/bin/activate
pip3 install -r requirements.txt
python36 -c "import nltk; nltk.download('wordnet')"
################################

python36 ExtractPotentialVNICs.py --FREQ_T 1 \
                                  --VNIC_DIR_PMI "./VNICs/PotentialVNICs_PMI_NoMin.csv" \
                                  --VNIC_DIR_LEX "./VNICs/PotentialVNICs_LEX_NoMin.csv" \
                                  --VNIC_DIR_SYN "./VNICs/PotentialVNICs_SYN_NoMin.csv" \
                                  --VNIC_DIR_OVA "./VNICs/PotentialVNICs_OVA_NoMin.csv"

deactivate
