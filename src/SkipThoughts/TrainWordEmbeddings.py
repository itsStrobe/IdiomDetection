"""
    File:   TrainWordEmbeddings
    Author: Jose Juan Zavala Iglesias
    Date:   21/05/2019

    Train Word Embedding Models with BNC XML Corpora and Store them for later use.
"""
import numpy as np
from Util import CorpusExtraction
from WordEmbeddings import Embeddings as SkipThoughts

model = None

## VOCABULARY INITIALIZATION ##
# Initializing Skip-Thoughts' Vocabulary
print("Initializing Model - NOT NEEDED: Using Pre-Trained Model ; Loading Model Instead")
model = SkipThoughts()

## MODEL TRAINING ##
# Train Skip-Thoughts - NOT NEEDED: Using Pre-Trained Model
print("Training Model - NOT NEEDED: Using Pre-Trained Model")

## MODEL TESTING ##
# TODO: Testing Skip-Thoughts
print("Testing Model")
featVector = model.GenerateFeatMatrix(np.array(["I am happy .", "I love everyone ."]))
print(featVector)
print(featVector.shape)

## SAVING MODELS ##
# TODO: Saving Skip-Thoughts - NOT NEEDED: Using Pre-Trained Model
print("Saving Model - NOT NEEDED: Using Pre-Trained Model")
