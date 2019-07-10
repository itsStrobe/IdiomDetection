#!/bin/bash
# Complete Run.
# Only run PrepareData.sh if absolutely necessary.

echo "RunAll.sh"

# ./PrepareData.sh

# ./FindVNICs.sh

./TrainAndGenerateEmbeddings.sh

./RunExperiments.sh
