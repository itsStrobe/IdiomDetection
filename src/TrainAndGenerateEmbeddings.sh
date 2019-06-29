#!/bin/bash

# ============================================================ #

# Word2Vec
cd ./Word2Vec
. venv/bin/activate

python3 TrainWordEmbeddings.py
python3 GenerateWordEmbeddings.py

deactivate
cd ..

# ============================================================ #

# Siamese CBOW
cd ./SiameseCBOW
. venv/bin/activate

python TrainWordEmbeddings.py
python GenerateWordEmbeddings.py

deactivate
cd ..

# ============================================================ #

# Skip-Thoughts
cd ./SkipThoughts
. venv/bin/activate

python TrainWordEmbeddings.py
python GenerateWordEmbeddings.py

deactivate
cd ..

# ============================================================ #

# ELMo
cd ./ELMo
. venv/bin/activate

python3 TrainWordEmbeddings.py
python3 GenerateWordEmbeddings.py

deactivate
cd ..
