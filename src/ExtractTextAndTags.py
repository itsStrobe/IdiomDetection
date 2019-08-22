"""
    File:   Util
    Author: Jose Juan Zavala Iglesias
    Date:   05/06/2019

    Extract the lemmatized text and POS Tags (C5 format) from the BNC XML Corpus.
"""
from Util import CorpusExtraction

CORPORA_DIR = "./Corpora/BNC XML/2554/download/Texts_CleanXML"
TEXT_SUFFIX  = "_CleanText_Lemma"
POS_SUFFIX   = "_PosTags"
GET_LEMMA    = True
GET_POS_TAGS = True

CorpusExtraction.ExtractCorpora(CORPORA_DIR, outDirSuffix=TEXT_SUFFIX, posDirSuffix=POS_SUFFIX, getLemma=GET_LEMMA, getPosTags=GET_POS_TAGS)
