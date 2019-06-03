"""
    File:   Util
    Author: Jose Juan Zavala Iglesias
    Date:   20/02/2019

    Generate Pre-Processed Corpora for Word Embedding Models Training
"""

from Util import CorpusExtraction

# BeautifulSoup Documentation on Tags: https://www.crummy.com/software/BeautifulSoup/bs4/doc/#tag

CORPORA_DIR = "./Corpora/BNC XML/2554/download/Texts/"
CORPORA_PRE = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K"]
INDEXED_SUF = "_idx"

# Extract Corpora as Sentences for Embedding Model Training
for prefix in CORPORA_PRE:
    print("Extracting Corpora:", prefix)
    corpora = CORPORA_DIR + prefix
    CorpusExtraction.SaveCorpora(corpora, prefix)
    CorpusExtraction.SaveCorpora(corpora, prefix, suffix=INDEXED_SUF, indexed=True)

print("Succesfully Extracted Corpora")
