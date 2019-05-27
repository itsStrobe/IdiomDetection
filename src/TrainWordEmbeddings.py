"""
    File:   TrainWordEmbeddings
    Author: Jose Juan Zavala Iglesias
    Date:   21/02/2019

    Train Word Embedding Models with BNC XML Corpora and Store them for later use.
"""
import numpy as np
from Util import CorpusExtraction
from WordEmbeddings import Word2Vec_Embeddings as W2V
from WordEmbeddings import SiameseCBOW_Embeddings as CBOW
from WordEmbeddings import ELMo_Embeddings as ELMo

CORPORA_PRE  = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K"]
MODEL_DIR    = "./models/"
MODEL_SUFFIX = "_ver1.model"

# Following values by King and Cook (2018)
VEC_SIZE = 300
WND_SIZE = 8
EPOCHS   = 10

train_W2V = False
train_SCB = False
train_SkT = False
train_ELM = True

model_W2V = None
model_SCB = None
model_SkT = None
model_ELM = None

# CorporaIterator = [sent for sent in CorpusExtraction.IterateOverCorpora(CORPORA_PRE)]

## VOCABULARY INITIALIZATION ##
# Initializing Word2Vec's Vocabulary
if (train_W2V): model_W2V = W2V(corpus=CorporaIterator, size=VEC_SIZE, window=WND_SIZE)
# Initializing Siamese CBOW's Vocabulary - NOT NEEDED: Using Pre-Trained Model ; Loading Model Instead
if (train_SCB): model_SCB = CBOW(pretrainned="./models/cosine_sharedWeights_adadelta_lr_1_noGradClip_epochs_2_batch_100_neg_2_voc_65536x300_noReg_lc_noPreInit_vocab_65535.end_of_epoch_2_p3.pkl")
# TODO: Initializing Skip-Thoughts' Vocabulary
# Initializing ELMo's Vocabulary - NOT NEEDED: Using Pre-Trained Tensorflow Hub Model ; Loading Model Instead
if (train_ELM): model_ELM = ELMo(hub_module="https://tfhub.dev/google/elmo/2")

## MODEL TRAINING ##
# Train Word2Vec
if (train_W2V): model_W2V.train(CorporaIterator, epochs=EPOCHS)
# Train Siamese CBOW - NOT NEEDED: Using Pre-Trained Model
# TODO: Train Skip-Thoughts
# Train ELMo - NOT NEEDED: Using Pre-Trained Hub Model

## MODEL TESTING ##
# Testing Word2Vec
if (train_W2V): print(model_W2V.GetMostSimilar("happy"))
# Testing Siamese CBOW
if (train_SCB): print(model_SCB.GetMostSimilar("happy"))
# TODO: Testing Skip-Thoughts
if (train_ELM): print(model_ELM.GenerateFeatMatrix(np.array(["I am sad .", "I hate everyone ."])))

## SAVING MODELS ##
# Saving Word2Vec
if (train_W2V): model_W2V.save(MODEL_DIR + "W2V" + MODEL_SUFFIX)
# Saving Siamese CBOW - NOT NEEDED: Using Pre-Trained Model
# TODO: Saving Skip-Thoughts
# Saving ELMo - NOT NEEDED: Using Pre-Trained Hub Model
