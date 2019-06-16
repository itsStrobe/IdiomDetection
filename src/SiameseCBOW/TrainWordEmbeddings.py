"""
    File:   TrainWordEmbeddings
    Author: Jose Juan Zavala Iglesias
    Date:   21/05/2019

    Train Word Embedding Models with BNC XML Corpora and Store them for later use.
"""
from WordEmbeddings import Embeddings as CBOW

PRETRAINNED_MODEL = "./models/cosine_sharedWeights_adadelta_lr_1_noGradClip_epochs_2_batch_100_neg_2_voc_65536x300_noReg_lc_noPreInit_vocab_65535.end_of_epoch_2.pickle"

model = None

## VOCABULARY INITIALIZATION ##
# Initializing Siamese CBOW's Vocabulary - NOT NEEDED: Using Pre-Trained Model ; Loading Model Instead
print("Initializing Model - NOT NEEDED: Using Pre-Trained Model ; Loading Model Instead")
model = CBOW(pretrainned=PRETRAINNED_MODEL)

## MODEL TRAINING ##
# Train Siamese CBOW - NOT NEEDED: Using Pre-Trained Model
print("Training Model - NOT NEEDED: Using Pre-Trained Model")

## MODEL TESTING ##
# Testing Siamese CBOW
print("Testing Model")
print(model.GetMostSimilar("happy"))

## SAVING MODELS ##
# Saving Siamese CBOW - NOT NEEDED: Using Pre-Trained Model
print("Saving Model - NOT NEEDED: Using Pre-Trained Model")
