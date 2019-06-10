"""
    File:   Util
    Author: Jose Juan Zavala Iglesias
    Date:   19/02/2019

    Util packages for file processing and generic applications.
"""

import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as Soup

class CorpusExtraction:

    """
    Reads a Corpus from the BNC XML Dataset format. Returns it as a Numpy of lists or a lists of lists.
    Each list represents a tokenized sentence.
    """
    @staticmethod
    def ReadCorpus(fileDir, asNumpy=True):
        tagRegexp = re.compile(r'<.*>')
        flatten = lambda l: [item for sublist in l for item in sublist]
        with open(fileDir, "r", encoding="utf_8") as corpus:
            corpusSoup = Soup(corpus, 'xml')
            sentTags   = corpusSoup.findAll('s')
            sent       = [flatten([token.text.split() for token in sentTag if tagRegexp.search(str(token))]) for sentTag in sentTags]

            if(asNumpy):
                return np.array(sent)
            else:
                return sent

    """
    Reads the Corpora in a given directory. Creates a map in which in it allocates each extracted corpus.
    """
    @staticmethod
    def ReadCorpora(rootDir):
        corpora = {}
        for root, _, files in os.walk(rootDir):
            if files == []:
                continue

            print("Extracting Corpora in:", root)
            for corpus in files:
                fileDir = os.path.join(root, corpus)
                print(fileDir)
                name, _ = os.path.splitext(corpus)
                corpora[name] = IO_Util.ReadCorpus(fileDir)

        return corpora

    """
    Saves a extracted corpora into a pickle file for quick access.
    """
    @staticmethod
    def SaveCorpora(rootDir, fileName):
        corporaName = 'data/' + fileName + '.pkl'
        corpora = IO_Util.ReadCorpora(rootDir)

        with open(corporaName, 'wb+') as file:
            pickle.dump(corpora, file, pickle.HIGHEST_PROTOCOL)

    """
    Loads a pre-saved corpora from a pickle file.
    """
    @staticmethod
    def LoadCorpora(fileName):
        corporaName = 'data/' + fileName + '.pkl'

        if(os.path.isfile(corporaName)):
            with open(corporaName, 'rb') as file:
                return pickle.load(file)

    @staticmethod
    def IterateOverCorpora(corporaDir, sentAsList=True):
        for corpora_name in corporaDir:
            print("Loading Corpora:", corpora_name)
            corpora = CorpusExtraction.LoadCorpora(corpora_name)
            for corpus in corpora:
                for sentence in corpora[corpus]:
                    if(sentAsList == False): ' '.join(sentence)
                    yield sentence    