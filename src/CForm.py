"""
    File:   CForm
    Author: Jose Juan Zavala Iglesias
    Date:   06/06/2019

    Implementation of CForm Model as described by Fazly et al. (2009)
"""

import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
import VNCPatternCounts

MAX_WINDOW = 5

class CForm(object):

    # INSTANCE FUNCTIONS
    def LoadModel(self, modelDir):
        with open(modelDir, 'rb') as file:
            self.model = pickle.load(file)

    def SaveModel(self, modelDir):
        # Create modelDir
        if not os.path.exists(os.path.dirname(modelDir)):
            try:
                os.makedirs(os.path.dirname(modelDir))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(modelDir, 'wb+') as file:
            pickle.dump(self.model, file, pickle.HIGHEST_PROTOCOL)

    def PatternZ_Score(self, verb, noun, pattern):
        vnc = (verb, noun)

        if(self.model is None):
            print("Pattern Counts are not loaded.")
            return cForms

        if(vnc not in self.model):
            print("VNC Not in Pattern Counts.")
            return cForms

        patterns = self.model[vnc]

        mean = np.mean(patterns[1:])
        std  = np.std(patterns[1:])

        return ((patterns[pattern] - mean) / std)


    def GetCanonicalForms(self, verb, noun, T=None):
        cForms = []

        if(T is None):
            T = self.T

        for pattern in range(1, 12):
            if(self.PatternZ_Score(verb, noun, pattern) > T):
                cForms.append(pattern)

        return cForms

    def IsCForm(self, verb, noun, sentence, posTags, T=None, max_window=MAX_WINDOW):
        cForms = self.GetCanonicalForms(verb, noun, T=T)

        for pattern in VNCPatternCounts.ExtractPatternsFromSentence(sentence, posTags, max_window=max_window):
            if((pattern[0] == verb) and (pattern[1] == noun) and (int(pattern[2]) in cForms)):
                return True

        return False


    def __init__(self, patternFileDir=None, modelDir=None, T=None):
        self.model = None
        self.T     = 1

        if(modelDir is not None):
            self.LoadModel(modelDir)
        elif(patternFileDir is not None):
            self.model = VNCPatternCounts.GenerateModelFromPatternFiles(patternFileDir)

        if(T is not None):
            self.T = T

