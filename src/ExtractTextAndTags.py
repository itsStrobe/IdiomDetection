"""
    File:   Util
    Author: Jose Juan Zavala Iglesias
    Date:   05/06/2019

    Extract the lemmatized text and POS Tags (C5 format) from the BNC XML Corpus.
"""
from Util import CorpusExtraction

CORPORA_ROOT = "./Corpora/BNC XML/2554/download/Texts"

CorpusExtraction.ExtractCorpora(CORPORA_ROOT)
