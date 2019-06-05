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
    def ReadCorpus(fileDir, asNumpy=True, indexed=False):
        tagRegexp = re.compile(r'<.*>')
        flatten = lambda l: [item for sublist in l for item in sublist]
        with open(fileDir, "r", encoding="utf_8") as corpus:
            corpusSoup = Soup(corpus, 'xml')
            sentTags   = corpusSoup.findAll('s')
            if(indexed):
                sent = {}
                for sentTag in sentTags:
                    if(sentTag['n'].isdigit()):
                        sent[int(sentTag['n'])] = flatten([tag.text.split() for tag in sentTag if tagRegexp.search(str(tag))])
            else:
                sent = [flatten([tag.text.split() for tag in sentTag if tagRegexp.search(str(tag))]) for sentTag in sentTags]
                if(asNumpy):
                    sent = np.array(sent)

        return sent

    """
    Reads the Corpora in a given directory. Creates a map in which in it allocates each extracted corpus.
    """
    @staticmethod
    def ReadCorpora(rootDir, indexed=False):
        corpora = {}
        for root, _, files in os.walk(rootDir):
            if files == []:
                continue

            print("Extracting Corpora in:", root)
            for corpus in files:
                fileDir = os.path.join(root, corpus)
                print(fileDir)
                name, _ = os.path.splitext(corpus)
                corpora[name] = CorpusExtraction.ReadCorpus(fileDir, indexed=indexed)

        return corpora

    """
    Saves a extracted corpora into a pickle file for quick access.
    """
    @staticmethod
    def SaveCorpora(rootDir, fileName, suffix='', indexed=False):
        corporaName = './data/' + fileName + suffix + '.pkl'
        corpora = CorpusExtraction.ReadCorpora(rootDir, indexed=indexed)

        with open(corporaName, 'wb+') as file:
            pickle.dump(corpora, file, pickle.HIGHEST_PROTOCOL)

    """
    Loads a pre-saved corpora from a pickle file.
    """
    @staticmethod
    def LoadCorpora(fileName, suffix=''):
        corporaName = './data/' + fileName + suffix + '.pkl'

        if(os.path.isfile(corporaName)):
            with open(corporaName, 'rb') as file:
                return pickle.load(file)

    @staticmethod
    def IterateOverCorpora(corporaDir, suffix=''):
        for corpora_name in corporaDir:
            print("Loading Corpora:", corpora_name)
            corpora = CorpusExtraction.LoadCorpora(corpora_name, suffix=suffix)
            for corpus in corpora:
                for sentence in corpora[corpus]:
                    yield sentence

class CorpusEdition:
    """
    Iterates over the lines of a Corpus from the BNC XML Dataset. Removes a list of patterns. 
    Prints the processed files into outFileDir
    """
    @staticmethod
    def RemovePatternsFromCorpus(inFileDir, outFileDir, patterns):
        with open(inFileDir, "r", encoding="utf_8") as inFile:
            if not os.path.exists(os.path.dirname(outFileDir)):
                try:
                    os.makedirs(os.path.dirname(outFileDir))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            with open(outFileDir, "w+", encoding="utf_8") as outFile:
                for line in inFile:
                    edited_line = line
                    for pattern in patterns:
                        edited_line = re.sub(pattern, "", edited_line)

                    outFile.write(edited_line)

    """
    Iterates over the Corpora in a given directory. Removes patters from all files.
    Prints the editted files into a new subdirectory specified by the split.
    """
    @staticmethod
    def RemovePatternsFromCorpora(rootDir, patterns, processedDirSuffix="_RemPatterns"):
        for root, _, files in os.walk(rootDir):
            if files == []:
                continue

            print("Extracting Corpora in:", root)
            for corpus in files:
                inFileDir  = os.path.join(root, corpus)
                outFileDir = inFileDir.replace(rootDir, rootDir + processedDirSuffix)
                print(inFileDir)
                name, _ = os.path.splitext(corpus)
                CorpusEdition.RemovePatternsFromCorpus(inFileDir, outFileDir, patterns)
