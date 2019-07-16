#!/bin/bash
# Train Models with Corpora and Generate Embeddings for VNC-Dataset and VNC-Candidates

echo "GenerateEmbeddings.sh"

# ============================================================ #

# Word2Vec
cd ./Word2Vec

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python3 venv && source venv/bin/activate
pip3 install -r requirements.txt
################################\

echo "Generating Word2Vec Embeddings"
python3 GenerateWordEmbeddings.py

deactivate
cd ..

# ============================================================ #

# Siamese CBOW
cd ./SiameseCBOW

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python3 venv && source venv/bin/activate
pip3 install -r requirements.txt
################################

echo "Generating SiameseCBOW Embeddings"
python GenerateWordEmbeddings.py

deactivate
cd ..

# ============================================================ #

# Skip-Thoughts
cd ./SkipThoughts

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python3 venv && source venv/bin/activate
pip3 install -r requirements.txt
################################

echo "Generating Skip-Thoughts Embeddings"
python GenerateWordEmbeddings.py

deactivate
cd ..

# ============================================================ #

# ELMo
cd ./ELMo

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python3 venv && source venv/bin/activate
pip3 install -r requirements.txt
################################

echo "Generating ELMo Embeddings"
python3 GenerateWordEmbeddings.py

deactivate
cd ..
