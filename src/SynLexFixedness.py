"""
    File:   SynLexFixedness
    Author: Jose Juan Zavala Iglesias
    Date:   13/06/2019

    Model to calculate Syntactic and Lexical Fixedness as described by Fazly et al. (2009)
"""

import nltk
import math
import pickle
import numpy as np
from Word2Vec.WordEmbeddings import Embeddings as w2v
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import lin_thesaurus as lin

import VNCPatternCounts

# Download Lin's Thesaurus
nltk.download('lin_thesaurus')

# CONSTANTS
K        = 50 # Number of similar verb/nouns for Lexical Fixedness equation. Base value based on work by Fazly et al. (2009)
LOG_BASE = 2
W2V_SIM  = 100 # Base number of W2V similar words; Increase if experiments prove necessary
ALPHA    = 0.6 # Base number for alpha parameter in Overall Fixedness equation. Base value based on work by Fazly et al. (2009)

class SynLexFixedness(object):

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

    def LoadW2VModel(self, w2vModelDir):
        self.w2vModel = w2v()
        self.w2vModel.load(w2vModelDir)

    # LEXICAL FIXEDNESS AUXILIARY METHODS #    

    def GetKMostSimilar(self, word, k=K, isNoun=True, useLin=True):
        kSimilar = []

        if(useLin):
            if(isNoun):
                fileid = 'simN.lsp'
            else:
                fileid = 'simV.lsp'

            simSet = list(lin.scored_synonyms(word, fileid=fileid))

            for token in list(simSet):
                if(len(kSimilar) >= k):
                    return kSimilar
                kSimilar.append(token[0].lower())
        else:
            if(self.w2vModel is None):
                print("W2V Model Not Loaded")
                return None

            if(isNoun):
                pos = wn.NOUN
            else:
                pos = wn.VERB

            # This method treats kSimilar as Set to avoid repeating tokens
            kSimilar = {word}

            for syn in wn.synsets(word, pos=pos):
                for l in syn.lemmas():
                    if(len(kSimilar) >= k):
                        return list(kSimilar)
                    kSimilar.add(l.name().lower())

            lemmatizer = WordNetLemmatizer()

            for simWord in self.w2vModel.GetMostSimilar(word, topN=W2V_SIM):
                if(len(kSimilar) >= k):
                    return list(kSimilar)

                lemWord = lemmatizer.lemmatize(simWord[0], pos=pos)
                kSimilar.add(lemWord.lower())

        # Ensure kSimilar is a list and not a Set
        kSimilar = list(kSimilar)

        while(len(kSimilar) < k):
            kSimilar.append("NOT_A_TOKEN")

        return kSimilar

    def GetKMostSimilarVNCs(self, verb, noun, vK=K, nK=K, useLin=True):
        simVerbs = self.GetKMostSimilar(verb, k=vK, isNoun=False, useLin=useLin)
        simNouns = self.GetKMostSimilar(noun, k=nK, isNoun=True, useLin=useLin)

        simVNCs = [(verb, noun)]

        for simVerb in simVerbs:
            simVNCs.append((simVerb, noun))

        for simNoun in simNouns:
            simVNCs.append((verb, simNoun))

        return simVNCs

    def CalcVerbNounCounts(self):
        if(self.model is None):
            print("VNC Model not loaded")
            return

        nounCounts = {}
        verbCounts = {}
        patCounts  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for vnc in self.model:
            verb  = vnc[0]
            noun  = vnc[1]
            count = self.model[vnc][0]

            if noun not in nounCounts:
                nounCounts[noun] = 0
            if verb not in verbCounts:
                verbCounts[verb] = 0

            nounCounts[noun] += count
            verbCounts[verb] += count
            for i in range(12):
                patCounts[i] += self.model[vnc][i]

        self.nounCounts = nounCounts
        self.verbCounts = verbCounts
        self.patCounts  = patCounts

    def PMI(self, verb, noun, logBase=LOG_BASE, verbose=False):
        if(self.model is None):
            print("VNC Model not loaded")
            return None
        if(self.nounCounts is None):
            print("Noun Counts not available. Run CalcVerbNounCounts()")
            return None
        if(self.verbCounts is None):
            print("Verb Counts not available. Run CalcVerbNounCounts()")
            return None

        # N_v+n       -> len(self.model)
        # f(V_r, N_t) -> self.model[(verb, noun)][0]
        # f(V_r,  * ) -> self.verbCounts[verb]
        # f( * , N_t) -> self.nounCounts[noun]

        if verb not in self.verbCounts:
            if(verbose): print("Verb <" + verb + "> not in Model")
            return 0

        if noun not in self.nounCounts:
            if(verbose): print("Noun <" + noun + "> not in Model")
            return 0

        if (verb, noun) not in self.model:
            if(verbose): print("VNC <"+ verb + ", " + noun +"> not in Model")
            return 0
        
        return math.log(((len(self.model) * self.model[(verb, noun)][0]) / (self.verbCounts[verb] * self.nounCounts[noun])), logBase)

    def Fixedness_Lex(self, verb, noun, vK=K, nK=K, logBase=LOG_BASE, useLin=True):
        if(K is not None):
            vK = K
            nK = K

        simSetPMIs = []
        simSet = self.GetKMostSimilarVNCs(verb, noun, vK=vK, nK=nK, useLin=useLin)

        for simV, simN in simSet:
            simSetPMIs.append(self.PMI(simV, simN, logBase=logBase))

        avgPMI = np.mean(simSetPMIs)
        stdPMI = np.std(simSetPMIs)

        return (self.PMI(verb, noun, logBase=logBase) - avgPMI) / stdPMI

    # SYNTACTIC FIXEDNESS AUXILIARY METHODS #

    def PatMLE(self, pat):
        if(self.patCounts is None):
            print("Pattern Counts not available. Run CalcVerbNounCounts()")
            return None
        return self.patCounts[pat] / self.patCounts[0]

    def PatPosProb(self, pat, verb, noun):
        if(self.model is None):
            print("VNC Model not loaded")
            return None

        vnc = (verb, noun)

        return self.model[vnc][pat] / self.model[vnc][0]

    def Fixedness_Syn(self, verb, noun, logBase=LOG_BASE):
        fixSyn = 0

        for pat in range(1, 12):
            posProb = self.PatPosProb(pat, verb, noun)
            patProb = self.PatMLE(pat)
            if(posProb/patProb > 0):
                fixSyn += posProb * math.log(posProb / patProb, logBase)

        return fixSyn

    def Fixedness_Overall(self, verb, noun, alpha=ALPHA, vK=K, nK=K, logBase=LOG_BASE, useLin=True):
        return alpha * self.Fixedness_Syn(verb, noun, logBase=logBase) + (1 - alpha) * self.Fixedness_Lex(verb, noun, vK=vK, nK=nK, logBase=logBase, useLin=useLin)

    def __init__(self, patternFileDir=None, modelDir=None, w2vModelDir=None, K=None):
        self.model      = None
        self.patCounts  = None
        self.nounCounts = None
        self.verbCounts = None
        self.w2vModel   = None

        if(modelDir is not None):
            self.LoadModel(modelDir)
            self.CalcVerbNounCounts()
        elif(patternFileDir is not None):
            self.model = VNCPatternCounts.GenerateModelFromPatternFiles(patternFileDir)
            self.CalcVerbNounCounts()

        if(w2vModelDir is not None):
            self.w2vModel = w2v()
            self.w2vModel.load(w2vModelDir)

        if(K is not None):
            self.K = K
