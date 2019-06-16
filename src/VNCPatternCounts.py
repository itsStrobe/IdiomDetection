"""
    File:   VNCPatternCounts
    Author: Jose Juan Zavala Iglesias
    Date:   06/06/2019

    VNCPattern counts for VNCPatternCounts Model and Syntactic Fixedness as described by Fazly et al. (2009)
"""

import os
import re
import sys
import pickle
import numpy as np
import pandas as pd

"""
Verb-Noun Patterns defined by Fazly et al. (2009)
Pattern 1   - VERB_ACT      NULL    NOUN_SG
Pattern 2   - VERB_ACT      a/an    NOUN_SG
Pattern 3   - VERB_ACT      the     NOUN_SG
Pattern 4   - VERB_ACT      DEM     NOUN_SG
Pattern 5   - VERB_ACT      POSS    NOUN_SG
Pattern 6   - VERB_ACT      NULL    NOUN_PL
Pattern 7   - VERB_ACT      the     NOUN_PL
Pattern 8   - VERB_ACT      DEM     NOUN_PL
Pattern 9   - VERB_ACT      POSS    NOUN_PL
Pattern 10  - VERB_ACT      OTHER   NOUN_SG/PL
Pattern 11  - NOUN_SG/PL    BE      VERB_PASS   # Rule modified with knowledge of passive form (Original: VERB_PASS ANY NOUN_SG/PL)

PUNC    ->  {., ,, ", /, \, :, ;, {, }, [, ], ~, `}
DEM     ->  {this, that, these, those}
POSS    ->  {my, your, his, her, its, our, their, 's, '}
OTHER   ->  TOKENS - (DEM U POSS U PUNC)
BE      ->  {be, been, being, am, are, is, was, were}

C5 Tags:
a/an     -> {AT0}
the      -> {AT0}
DEM      -> {DT0}
POSS     -> {DPS}
PUNC     -> {PUL, PUN, PUQ, PUR}
OTHER    -> U - {PUNC} # Check for this case after all other rules were rejected
BE       -> {VBB, VBD, VBI, VBN, VBZ}
VERB_ACT -> {VBB, VBD, VBG, VBI, VBN, VBZ, VDB, VDD, VDG, VDI, VDN, VDZ, VHB, VHD, VHG, VHI, VHN, VHZ, VM0, VVB, VVD, VVG, VVI, VVN, VVZ}
VERB_PASS-> {VBN, VDN, VHN, VVN} # Past participle tense of verbs
NOUN_SG  -> {NN0, NN1}
NOUN_PL  -> {NN0, NN2}

Notes:
>NN0 is being taken in both SG and PL rules. For patterns that match in both singular/plural
 cases (e.g. Patterns 3 and 7) both pattern counts will be registered.
>NP0 are being considered only as singular, since according to http://www.natcorp.ox.ac.uk/docs/URG/posguide.html#m2np0,
 plural proper nouns are "a comprarative rarity".
"""

# CONSTANTS
A_AN_THE = {'AT0'} # A/AN and THE combined into single set since they share C5 Tag
DEM      = {'DT0'}
POSS     = {'DPS'}
PUNC     = {'PUL', 'PUN', 'PUQ', 'PUR'}
BE       = {'VBB', 'VBD', 'VBI', 'VBN', 'VBZ'}
VERB_ACT = {'VBB', 'VBD', 'VBG', 'VBI', 'VBN', 'VBZ', 'VDB', 'VDD', 'VDG', 'VDI', 'VDN', 'VDZ', 'VHB', 'VHD', 'VHG', 'VHI', 'VHN', 'VHZ', 'VM0', 'VVB', 'VVD', 'VVG', 'VVI', 'VVN', 'VVZ'}
VERB_PASS= {'VBN', 'VDN', 'VHN', 'VVN'}
NOUN_SG  = {'NN0', 'NN1'}
NOUN_PL  = {'NN0', 'NN2'}

MAX_WINDOW = 5

# CLASS FUNCTIONS

"""
Finds the next noun in setence. Returns Integer if any and None if none.
"""
def FindNextNoun(idx, posTags):
    while(idx < len(posTags)):
        if(posTags[idx] in NOUN_SG or posTags[idx] in NOUN_PL):
            return idx
        idx += 1

    return None

"""
Disambiguation for similar patterns that have differences in cardinality.
"""
def SimilarPatternDesambiguation(nounTag, pat_SG, pat_PL, verb, noun):
    patterns = []

    if(nounTag in NOUN_SG and nounTag in NOUN_PL):
        patterns.append([verb, noun, pat_SG])
        patterns.append([verb, noun, pat_PL])
        return patterns
    if(nounTag in NOUN_SG):
        patterns.append([verb, noun, pat_SG])
        return patterns
    if(nounTag in NOUN_PL):
        patterns.append([verb, noun, pat_PL])
        return patterns

"""
Finds the fitting patterns (1-10) for a VNC.
"""
def FindPattern_1_10(verbPos, nounPos, sentence, posTags, max_window=MAX_WINDOW):
    patternLength = nounPos - verbPos
    patterns      = []

    if(patternLength >= max_window):
        return patterns # Outside set window

    # Check for patterns 1 and 6
    if(patternLength == 1):
        return SimilarPatternDesambiguation(posTags[nounPos], '1', '6', sentence[verbPos], sentence[nounPos])

    # Check for punctuation marks that interrupt pattern
    # This solves the last condition for Pattern 10
    for idx in range(verbPos, nounPos + 1):
        if(posTags[idx] in PUNC):
            return patterns

    # Check for pattern 2, 3, and 7
    for idx in range(verbPos, nounPos + 1):
        if(posTags[idx] in A_AN_THE):
            if(sentence[idx] == 'a' or sentence[idx] == 'an'):
                patterns.append([sentence[verbPos], sentence[nounPos], '2'])
                return patterns

            if(sentence[idx] == 'the'):
                return SimilarPatternDesambiguation(posTags[nounPos], '3', '7', sentence[verbPos], sentence[nounPos])

    # Check for pattern 4 and 8
    for idx in range(verbPos, nounPos + 1):
        if(posTags[idx] in DEM):
            return SimilarPatternDesambiguation(posTags[nounPos], '4', '8', sentence[verbPos], sentence[nounPos])

    # Check for pattern 5 and 9
    for idx in range(verbPos, nounPos + 1):
        if(posTags[idx] in POSS):
            return SimilarPatternDesambiguation(posTags[nounPos], '5', '9', sentence[verbPos], sentence[nounPos])

    # Check for pattern 10 - Since PUNCs were already checked, just return Pattern 10
    patterns.append([sentence[verbPos], sentence[nounPos], '10'])
    return patterns


def IsPuncInRange(range):
    for elem in range:
        if(elem in PUNC):
            return True

    return False


def IsBeInRange(range):
    for elem in range:
        if(elem in BE):
            return True

    return False

"""
Finds the fitting patterns (11) for a VNC.
"""
def FindPattern_11(nounPos, sentence, posTags, max_window=MAX_WINDOW):
    # Window smaller that required for Pattern 11 to occur.
    if(max_window < 3):
        return []

    if((nounPos + (max_window)) > len(sentence)):
        rightEdge = len(sentence)
    else:
        rightEdge = max_window      

    for verbPos in range(nounPos + 2, rightEdge):
        # PUNC breaks pattern
        if(IsPuncInRange(posTags[(nounPos + 1):verbPos])):
            return []

        if(posTags[verbPos] in VERB_PASS and IsBeInRange(posTags[(nounPos + 1):verbPos])):
            return [[sentence[verbPos], sentence[nounPos], '11']]

    return []

"""
Given a Sentence and its sequence of POS Tags (C5 Format), it extracts the VNCs and Pattern numbers.
"""
def ExtractPatternsFromSentence(sentence, posTags, max_window=MAX_WINDOW):
    if(isinstance(sentence, str)):
        sentence = sentence.split()
    if(isinstance(posTags, str)):
        posTags = posTags.split()

    patterns = []

    # Inconsistencies between Words and Tags - Ignore Sentence
    if(len(sentence) != len(posTags)):
        return patterns

    for idx in range(len(sentence)):
        # Pattern 1-10
        if(posTags[idx] in VERB_ACT):
            nextNoun = FindNextNoun(idx, posTags)
            while(nextNoun is not None and (nextNoun - idx) < max_window):
                patterns += FindPattern_1_10(idx, nextNoun, sentence, posTags, max_window=max_window)
                nextNoun = FindNextNoun(nextNoun + 1, posTags)

        # Pattern 11
        if(posTags[idx] in NOUN_SG or posTags[idx] in NOUN_PL):
            patterns += FindPattern_11(idx, sentence, posTags, max_window=max_window)

    return patterns

"""
Finds all VNC patterns in a Corpus given a sentence file and a tag file.
Outputs findings into a new directory.
"""
def ExtractPatternsFromCorpus(inFileDir, posFileDir, outFileDir, max_window=MAX_WINDOW):
    # Create outFileDir
    if not os.path.exists(os.path.dirname(outFileDir)):
        try:
            os.makedirs(os.path.dirname(outFileDir))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    patterns = []

    with open(inFileDir, "r", encoding="utf_8") as inFile:
        with open(posFileDir, "r", encoding="utf_8") as posFile:
            with open(outFileDir, "w+", encoding="utf_8") as outFile:
                for (sentence, posTags) in zip(inFile, posFile):
                    patterns += ExtractPatternsFromSentence(sentence, posTags, max_window=max_window)
                for pattern in patterns:
                    outFile.write(' '.join(pattern) + '\n')

"""
Envelope function for ExtractPatternsFromCorpus.
Iterates over all Corpus in the Corpora and writes the pattern files for each Corpus.
"""
def ExtractPatternsFromCorpora(corporaTextRootDir, outRootDir, cleanTextSuffix="_CleanText", posTagsTextSuffix="_PosTags", max_window=MAX_WINDOW):
    for root, _, files in os.walk(corporaTextRootDir):
        if files == []:
            continue

        print("Extracting Corpora in:", root)
        for corpus in files:
            inFileDir = os.path.join(root, corpus)

            inTextFileDir   = inFileDir.replace(corporaTextRootDir, corporaTextRootDir + cleanTextSuffix).replace('.xml', '.txt')
            inPosTagFileDir = inFileDir.replace(corporaTextRootDir, corporaTextRootDir + posTagsTextSuffix).replace('.xml', '.txt')
            outFileDir      = inFileDir.replace(corporaTextRootDir, outRootDir).replace('.xml', '.txt')

            print(inFileDir)
            ExtractPatternsFromCorpus(inTextFileDir, inPosTagFileDir, outFileDir, max_window=max_window)

"""
Gets all the VNC Pattern counts into a Dictionary.
Each VNC uses an array in which:
    Position  0    -> Total Number of Counts of VNC
    Positions 1-11 -> Pattern Count
"""
def GenerateModelFromPatternFile(patternFileDir, model={}):
    # Position  0    -> Total Number of Counts of VNC
    # Positions 1-11 -> Pattern Count
    defPatCount = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    with open(patternFileDir, "r", encoding="utf_8") as patternFile:
        for line in patternFile:
            pattern = line.split()

            vnc = (pattern[0], pattern[1])
            pat = int(pattern[2]) # Array indexing starts in 0

            if vnc not in model:
                model[vnc] = defPatCount.copy()

            model[vnc][0]   += 1
            model[vnc][pat] += 1

    return model

"""
Envelope function for GenerateModelFromPatternFile
Iterates over all Pattern Files and saves the VNC Pattern.
"""
def GenerateModelFromPatternFiles(patternFilesRoot):
    model = {}

    for root, _, files in os.walk(patternFilesRoot):
        if files == []:
            continue

        print("Extracting Corpora in:", root)
        for corpus in files:
            patternFileDir = os.path.join(root, corpus)

            print(patternFileDir)
            model = GenerateModelFromPatternFile(patternFileDir, model=model)

    return model

