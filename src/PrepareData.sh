#!/bin/bash

# Prepare the BNC XML for processing

python3 CleanBNC.py
python3 ExtractCorpora.py
python3 ExtractTextAndTags.py

# Extract the Different Patterns Inside the Processed Corpus

python3 ExtractPatternsInCorpora.py

# Extract the Target Sentences in the VNC Tokens Dataset

python3 ExtractTargetSentences.py

# Extract the Potential VNICs based on PMI

python3 ExtractPotentialVNICs.py
