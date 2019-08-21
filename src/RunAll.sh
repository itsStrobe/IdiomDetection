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

./TrainEmbeddings.sh

./FindVNICs.sh

./GenerateEmbeddings.sh

./TrainSVMs.sh

./GenerateSilverStandard.sh

./RunExperiments.sh
