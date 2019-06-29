"""
    File:   Util
    Author: Jose Juan Zavala Iglesias
    Date:   20/05/2019

    Generate Pre-Processed Corpora for Word Embedding Models Training
"""

import os

from Util import CorpusExtraction

# BeautifulSoup Documentation on Tags: https://www.crummy.com/software/BeautifulSoup/bs4/doc/#tag

CORPORA_DIR = "./Corpora/BNC XML/2554/download/Texts_CleanXML"
CORPORA_PRE = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K"]
INDEXED_SUF = "_idx"
POSTAGS_SUF = "_posTags"
LEMMAS_SUF  = "_lemmas"

# Extract Corpora as Sentences for Embedding Model Training
for prefix in CORPORA_PRE:
    print("Extracting Corpora:", prefix)
    corpora = os.path.join(CORPORA_DIR, prefix)
    print("Extracting Original Sentences")
    CorpusExtraction.SaveCorpora(corpora, prefix)
    print("Extracting Original Sentences - Indexed")
    CorpusExtraction.SaveCorpora(corpora, prefix, suffix=INDEXED_SUF, indexed=True)
    print("Extracting Sentences POS Tags - Indexed")
    CorpusExtraction.SaveCorpora(corpora, prefix, suffix=POSTAGS_SUF, indexed=True, posTags=True)
    print("Extracting Lemmatized Sentences - Indexed")
    CorpusExtraction.SaveCorpora(corpora, prefix, suffix=LEMMAS_SUF , indexed=True, lemmas=True)

print("Succesfully Extracted Corpora")
