"""
    File:   Util
    Author: Jose Juan Zavala Iglesias
    Date:   19/05/2019

    Util packages for file processing and generic applications.
"""

import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as Soup

ID_TAG    = 'n'
POS_TAG   = 'c5'
LEMMA_TAG = 'lemma'
TEXT_TAGS = ['w']

class CorpusExtraction:

    """
    Extracts the texts of a corpus either in the present state or in a lemmatized state.
    It has the posibility of also extracting the tags.
    """
    @staticmethod
    def ExtractCorpus(inFileDir, outFileDir, getLemma=True, posFileDir=None, hasMW=False, tags=TEXT_TAGS):
        if(hasMW):
            print("Not implemented")
            raise NotImplementedError

        # Create outFileDir
        if not os.path.exists(os.path.dirname(outFileDir)):
            try:
                os.makedirs(os.path.dirname(outFileDir))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # Create posFileDir
        if(posFileDir != None):
            if not os.path.exists(os.path.dirname(posFileDir)):
                try:
                    os.makedirs(os.path.dirname(posFileDir))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

        with open(inFileDir, "r", encoding="utf_8") as inFile:
            with open(outFileDir, "w+", encoding="utf_8") as outFile:
                fileSoup = Soup(inFile, 'xml')
                sentTags = fileSoup.findAll('s')
                print('-')
                print(sentTags)

                if(posFileDir != None):
                    with open(posFileDir, "w+", encoding="utf_8") as posFile:
                        for sentTag in sentTags:
                            sentence = ""
                            posTags  = ""
                            # Implement hasMW exception here:
                            wordTags = sentTag.findAll(tags)
                            for wordTag in wordTags:
                                if(wordTag.name == 'w' and getLemma):
                                    sentence = sentence + wordTag[LEMMA_TAG].strip().replace(" ", "_") + " "
                                else:
                                    sentence = sentence + wordTag.text.strip().replace(" ", "_") + " "

                                posTags = posTags + wordTag[POS_TAG].strip() + " "

                            sentence = sentence.strip() + '\n'
                            posTags  = posTags.strip() + '\n'

                            outFile.write(sentence)
                            posFile.write(posTags)
                else:
                    for sentTag in sentTags:
                        sentence = ""
                        # Implement hasMW exception here:
                        for wordTag in sentTag.findAll(tags):
                            if(wordTag.name == 'w' and getLemma):
                                sentence = sentence + wordTag[LEMMA_TAG].strip().replace(" ", "_") + " "
                            else:
                                sentence = sentence + wordTag.text.strip().replace(" ", "_") + " "

                        sentence = sentence.strip()

                        outFile.write(sentence)

    """
    Wrapper method for ExtractCorpus for iterating over the whole corpora.
    """
    @staticmethod
    def ExtractCorpora(rootDir, outDirSuffix="_CleanText", posDirSuffix="_PosTags", getLemma=True, getPosTags=True, hasMW=False, tags=TEXT_TAGS):
        for root, _, files in os.walk(rootDir):
            if files == []:
                continue

            print("Extracting Corpora in:", root)
            for corpus in files:
                inFileDir = os.path.join(root, corpus)
                outFileDir = inFileDir.replace(rootDir, rootDir + outDirSuffix).replace(".xml", ".txt")
                if(getPosTags):
                    posFileDir = inFileDir.replace(rootDir, rootDir + posDirSuffix).replace(".xml", ".txt")
                else:
                    posFileDir = None
                print(inFileDir)
                CorpusExtraction.ExtractCorpus(inFileDir, outFileDir, getLemma=getLemma, posFileDir=posFileDir, hasMW=hasMW, tags=tags)

    """
    Reads a Corpus from the BNC XML Dataset format. Returns it as a Numpy of lists or a lists of lists.
    Each list represents a tokenized sentence.
    """
    @staticmethod
    def ReadCorpus(fileDir, asNumpy=True, indexed=False, posTags=False, lemmas=False):
        tags=TEXT_TAGS
        tagRegexp = re.compile(r'<.*>')
        flatten = lambda l: [item for sublist in l for item in sublist]
        with open(fileDir, "r", encoding="utf_8") as corpus:
            corpusSoup = Soup(corpus, 'xml')
            sentTags   = corpusSoup.findAll('s')
            if(indexed):
                sent = {}
                for sentTag in sentTags:
                    if(sentTag[ID_TAG].isdigit()):
                        if(posTags):
                            sent[int(sentTag[ID_TAG])] = [wordTag[POS_TAG] for wordTag in sentTag.findAll(tags) if tagRegexp.search(str(wordTag))]
                        elif(lemmas):
                            sent[int(sentTag[ID_TAG])] = [wordTag[LEMMA_TAG] for wordTag in sentTag.findAll(tags) if tagRegexp.search(str(wordTag))]
                        else:
                            sent[int(sentTag[ID_TAG])] = flatten([tag.text.split() for tag in sentTag.findAll(tags) if tagRegexp.search(str(tag))])
            else:
                if(posTags):
                    sent = [wordTag[POS_TAG] for wordTag in sentTag.findAll(tags) if tagRegexp.search(str(wordTag)) for sentTag in sentTags]
                elif(lemmas):
                    sent = [wordTag[LEMMA_TAG] for wordTag in sentTag.findAll(tags) if tagRegexp.search(str(wordTag)) for sentTag in sentTags]
                else:
                    sent = [flatten([tag.text.split() for tag in sentTag.findAll(tags) if tagRegexp.search(str(tag))])  for sentTag in sentTags]

                if(asNumpy):
                    sent = np.array(sent)

        return sent

    """
    Reads the Corpora in a given directory. Creates a map in which in it allocates each extracted corpus.
    """
    @staticmethod
    def ReadCorpora(rootDir, indexed=False, posTags=False, lemmas=False):
        corpora = {}
        print(rootDir)
        for root, _, files in os.walk(rootDir):
            if files == []:
                continue

            print("Extracting Corpora in:", root)
            for corpus in files:
                fileDir = os.path.join(root, corpus)
                print(fileDir)
                name, _ = os.path.splitext(corpus)
                corpora[name] = CorpusExtraction.ReadCorpus(fileDir, indexed=indexed, posTags=posTags, lemmas=lemmas)

        return corpora

    """
    Saves a extracted corpora's Pos Tags into a pickle file for quick access.
    """
    @staticmethod
    def SaveCorpora(rootDir, fileName, suffix='', indexed=False, posTags=False, lemmas=False, corporaDir='./data'):
        corporaName = os.path.join(corporaDir, fileName + suffix + '.pkl')
        corpora = CorpusExtraction.ReadCorpora(rootDir, indexed=indexed, posTags=posTags, lemmas=False)

        with open(corporaName, 'wb+') as file:
            pickle.dump(corpora, file, pickle.HIGHEST_PROTOCOL)

    """
    Loads a pre-saved corpora from a pickle file.
    """
    @staticmethod
    def LoadCorpora(fileName, suffix='', corporaDir="./data"):
        corporaName = os.path.join(corporaDir, fileName + suffix + '.pkl')

        if(os.path.isfile(corporaName)):
            with open(corporaName, 'rb') as file:
                return pickle.load(file)

    """
    Creates an iterator for reading a Corpora.
    """
    @staticmethod
    def IterateOverCorpora(corporaPrefix, suffix='', corporaDir='./data/', asLower=False):
        for corpora_name in corporaPrefix:
            print("Loading Corpora:", corpora_name)
            corpora = CorpusExtraction.LoadCorpora(corpora_name, suffix=suffix, corporaDir=corporaDir)
            for corpus in corpora:
                for sentence in corpora[corpus]:
                    if(asLower):
                        for idx in range(len(sentence)):
                            sentence[idx] = sentence[idx].lower()
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
                CorpusEdition.RemovePatternsFromCorpus(inFileDir, outFileDir, patterns)
