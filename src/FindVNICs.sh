#!/bin/bash
# Extract the Different Patterns Inside the Processed Corpus

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python3 venv && source venv/bin/activate
pip3 install -r requirements.txt
python3 -c "import nltk; nltk.download()"
################################

echo "FindVNICs.sh"

echo "Extracting Patterns from Corpora"
python3 ExtractPatternsInCorpora.py

# Extract the Potential VNICs based on PMI

echo "Extracting Potential VNICs from Corpora"
python3 ExtractPotentialVNICs.py

echo "Extracting Potential VNICs Instances from Corpora"
python3 ExtractVNICsInstances.py

echo "Extracting Potential VNICs Sentences from Corpora"
python3 ExtractCandidateSentences.py --NORM_FIX --USE_LIN

deactivate
