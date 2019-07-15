#!/bin/bash
# Train Models with Corpora and Generate Embeddings for VNC-Dataset and VNC-Candidates

echo "TrainAndGenerateEmbeddings.sh"

# ============================================================ #

# Word2Vec
cd ./Word2Vec

# - VIRTUAL EVIRONMENT SETUP - #
virtualenv -p python3 venv && source venv/bin/activate
pip3 install -r requirements.txt
python3 -c "import nltk; nltk.download()"
################################

echo "Training Word2Vec"
#python3 TrainWordEmbeddings.py

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
python3 -c "import nltk; nltk.download()"
################################

echo "Training SiameseCBOW"
#python TrainWordEmbeddings.py

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
python3 -c "import nltk; nltk.download()"
################################

echo "Training Skip-Thoughts"
python TrainWordEmbeddings.py

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
python3 -c "import nltk; nltk.download()"
################################

echo "Training ELMo"
python3 TrainWordEmbeddings.py

echo "Generating ELMo Embeddings"
python3 GenerateWordEmbeddings.py

deactivate
cd ..
