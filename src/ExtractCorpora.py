"""
    File:   Util
    Author: Jose Juan Zavala Iglesias
    Date:   20/02/2019

    Generate Pre-Processed Corpora for Word Embedding Models Training
"""

from Util import CorpusExtraction

CORPORA_DIR = "./Corpora/BNC XML/2554/download/Texts/"
CORPORA_PRE = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K"]

for prefix in CORPORA_PRE:
    print("Extracting Corpora:", prefix)
    corpora = CORPORA_DIR + prefix
    CorpusExtraction.SaveCorpora(corpora, prefix)

print("Succesfully Extracted Corpora")
