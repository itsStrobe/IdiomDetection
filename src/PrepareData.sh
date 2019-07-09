#!/bin/bash
# Prepare the BNC XML for processing

echo "PrepareData.sh"

echo "Cleaning BNC"
# python3 CleanBNC.py

echo "Extracting Sentences from Corpora"
python3 ExtractCorpora.py

echo "Extracting Text and Tags from Corpora"
python3 ExtractTextAndTags.py

echo "Extract Instances of VNC-Dataset from Corpora"
python3 ExtractTargetSentences.py
