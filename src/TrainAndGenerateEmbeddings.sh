#!/bin/bash
# Train Models with Corpora and Generate Embeddings for VNC-Dataset and VNC-Candidates

echo "TrainAndGenerateEmbeddings.sh"

# ============================================================ #

# Word2Vec
cd ./Word2Vec
. venv/bin/activate

echo "Training Word2Vec"
#python3 TrainWordEmbeddings.py

echo "Generating Word2Vec Embeddings"
python3 GenerateWordEmbeddings.py

deactivate
cd ..

# ============================================================ #

# Siamese CBOW
cd ./SiameseCBOW
. venv/bin/activate

echo "Training SiameseCBOW"
#python TrainWordEmbeddings.py

echo "Generating SiameseCBOW Embeddings"
python GenerateWordEmbeddings.py

deactivate
cd ..

# ============================================================ #

# Skip-Thoughts
cd ./SkipThoughts
. venv/bin/activate

echo "Training Skip-Thoughts"
#python TrainWordEmbeddings.py

echo "Generating Skip-Thoughts Embeddings"
python GenerateWordEmbeddings.py

deactivate
cd ..

# ============================================================ #

# ELMo
cd ./ELMo
. venv/bin/activate

echo "Training ELMo"
#python3 TrainWordEmbeddings.py

echo "Generating ELMo Embeddings"
python3 GenerateWordEmbeddings.py

deactivate
cd ..
