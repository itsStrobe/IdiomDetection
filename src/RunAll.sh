#!/bin/bash
# FILE: RunAll.sh 
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu.q
#$ -l gpu=1
source /usr/local/gpuallocation.sh

echo "RunAll.sh"

./PrepareData.sh

./FindVNICs.sh

./TrainAndGenerateEmbeddings.sh

./RunExperiments.sh
