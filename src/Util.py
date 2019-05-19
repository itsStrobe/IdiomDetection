"""
    File:   Util
    Author: Jose Juan Zavala Iglesias
    Date:   19/02/2019

    Util packages for file processing and generic applications.
"""

import sys
import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as Soup

class IO_Util:

    """
    Reads a Corpus from the BNC XML Dataset format. Returns it as a Numpy of lists or a lists of lists.
    Each list represents a tokenized sentence.
    """
    @staticmethod
    def ReadCorpus(fileDir, asNumpy=True):
        with open(fileDir, "r", encoding="utf_8") as corpus:
            corpusSoup = Soup(corpus, 'xml')
            sentTags   = corpusSoup.findAll('s')
            sent       = [[x.text.strip() for x in sentTag] for sentTag in sentTags]

            if(asNumpy):
                return np.array(sent)
            else:
                return sent

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
