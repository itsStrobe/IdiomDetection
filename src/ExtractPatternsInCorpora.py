"""
    File:   ExtractPatternsInCorpora
    Author: Jose Juan Zavala Iglesias
    Date:   07/06/2019

    Extract the Verb-Noun Combinations and Patterns from Corpora
"""

from CForm import CForm
import VNCPatternCounts

CORPORA_DIR   ="./Corpora/BNC XML/2554/download/Texts_CleanXML"
TEXT_SUFFIX   = "_CleanText_Lemma"
TAGS_SUFFIX   = "_PosTags"
ROOT_PATTERNS = "./Patterns"
PAT_MODEL     = "./PatternCounts/PatternCounts_130619.pickle"

MAX_WINDOW = 7

VNCPatternCounts.ExtractPatternsFromCorpora(CORPORA_DIR, ROOT_PATTERNS, cleanTextSuffix=TEXT_SUFFIX, posTagsTextSuffix=TAGS_SUFFIX, max_window=MAX_WINDOW)

cForm_model = CForm(patternFileDir=ROOT_PATTERNS)
cForm_model.SaveModel(PAT_MODEL)
