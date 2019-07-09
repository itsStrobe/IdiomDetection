#!/bin/bash
# Extract the Different Patterns Inside the Processed Corpus

echo "FindVNICs.sh"

echo "Extracting Patterns from Corpora"
python3 ExtractPatternsInCorpora.py

# Extract the Potential VNICs based on PMI

echo "Extracting Potential VNICs from Corpora"
python3 ExtractPotentialVNICs.py

echo "Extracting Potential VNICs Instanecs from Corpora"
python3 ExtractVNICsInstances.py

echo "Extracting Potential VNICs Sentences from Corpora"
python3 ExtractCandidateSentences.py
