#!/bin/bash

# Prepare the BNC XML for processing

python3 CleanBNC.py
python3 ExtractCorpora.py
python3 ExtractTextAndTags.py
python3 ExtractTargetSentences.py
