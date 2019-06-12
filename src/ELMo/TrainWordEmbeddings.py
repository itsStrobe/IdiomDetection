"""
    File:   TrainWordEmbeddings
    Author: Jose Juan Zavala Iglesias
    Date:   21/05/2019

    Train Word Embedding Models with BNC XML Corpora and Store them for later use.
"""
import numpy as np
from WordEmbeddings import Embeddings as ELMo

HUB_MODULE = "https://tfhub.dev/google/elmo/2"

model = None

## VOCABULARY INITIALIZATION ##
# Initializing ELMo's Vocabulary - NOT NEEDED: Using Pre-Trained Tensorflow Hub Model ; Loading Model Instead
print("Initializing Model - NOT NEEDED: Using Pre-Trained Model ; Loading Model Instead")
model = ELMo(hub_module="https://tfhub.dev/google/elmo/2")

## MODEL TRAINING ##
# Train ELMo - NOT NEEDED: Using Pre-Trained Hub Model
print("Training Model - NOT NEEDED: Using Pre-Trained Model")

## MODEL TESTING ##
# Testing ELMo
print("Testing Model")
featVector = model.GenerateFeatMatrix(np.array(["I am happy .", "I love everyone ."])) 
print(featVector)
print(featVector.shape)

## SAVING MODELS ##
# Saving ELMo - NOT NEEDED: Using Pre-Trained Hub Model
print("Saving Model - NOT NEEDED: Using Pre-Trained Model")
