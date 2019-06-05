"""
    File:   RemoveMultiWordTags
    Author: Jose Juan Zavala Iglesias
    Date:   05/06/2019

    Remove all patterns of Multiwords (MW's) tags in the BNC XML Corpora.
"""

from Util import CorpusEdition

CORPORA_ROOT = "./Corpora/BNC XML/2554/download/Texts"
MW_STR_REGEX = r"<mw c5=\"(AJ0|CRD|ORD|AV0|AVQ|AT0|CJC|CJS|CJT|ITJ|PRP|DTQ|PNX|NN0|NN1|NN2)\">"
MW_END_REGEX = r"</mw>"

CorpusEdition.RemovePatternsFromCorpora(CORPORA_ROOT, [MW_STR_REGEX, MW_END_REGEX], processedDirSuffix="_NoMW")
