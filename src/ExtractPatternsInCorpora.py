"""
    File:   ExtractPatternsInCorpora
    Author: Jose Juan Zavala Iglesias
    Date:   07/06/2019

    Extract the Verb-Noun Combinations and Patterns from Corpora
"""

import VNCPatternCounts

ROOT_COPORA   ="./Corpora/BNC XML/2554/download/Texts"
TEXT_SUFFIX   = "_CleanText"
TAGS_SUFFIX   = "_PosTags"
ROOT_PATTERNS = "./Patterns"

MAX_WINDOW = 5

VNCPatternCounts.ExtractPatternsFromCorpora(ROOT_COPORA, ROOT_PATTERNS, cleanTextSuffix=TEXT_SUFFIX, posTagsTextSuffix=TAGS_SUFFIX, max_window=MAX_WINDOW)
