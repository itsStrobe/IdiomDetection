'''
Copyright 2016 Tom Kenter

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations
under the License.
'''

import glob
import os
import codecs
import numpy as np
import sys

sys.path.append("../siamese-skipgram")
import vocabUtils

class InexIterator:
  def __init__(self, sInexDir, sGlobPattern="*", bDontLowercase=False,
               bRandom=False, bSingleSentence=False, bVerbose=False, sName=''):
    '''
    The INEX directory provided here is supposed to contain directories, each
    of which should contain files in paragraph format:

    0<TAB>sentence1 paragraph1
    0<TAB>sentence2 paragraph1
    1<TAB>sentence1 paragraph2
    1<TAB>sentence2 paragraph2
    0<TAB>sentence1 paragraph3
    0<TAB>...

    What the iterator returns is a tuple:

    (sentence n, sentence n-1, sentence n+1),

    where sentence n and sentence n +/- 1 belong to the same paragraph.    

    In single sentence mode, and also in random mode, it will return a single
    (random) sentence
    '''
    if os.path.isdir(sInexDir):
      self.sInexDir = sInexDir
    else:
      print >>sys.stderr, "[InexIterator ERROR] '%s' is not a directory" % \
        sInexDir
      exit(1)
    self.iLines = 0
    self.bRandom = bRandom
    self.bSingleSentence = bSingleSentence
    self.bDontLowercase = bDontLowercase

    # You can use this pattern to make selection of INEX directories
    # want to consider. The directories are selected by calling:
    # 
    # glob.glob(sInexDIR/<GLOB_PATTERN>)
    #
    self.sGlobPattern = sGlobPattern

    self.bVerbose = bVerbose

    self.sName = sName
    
    self.sCurrentFile = None
    self.sPreviousSentence = None
    self.sPreviousPreviousSentence = None
    self.sPreviousParagraph = None
    self.iCurrentYieldIndex = -1

  def nextDir(self):
    if self.bRandom:
      while 1: # In the random case, we keep on going forever...
        if self.iCurrentYieldIndex == (len(self.aAllDirs) - 1):
          # We've reached the end: reset
          self.iCurrentYieldIndex = -1
          np.random.shuffle(self.aAllDirs)

        self.iCurrentYieldIndex += 1
        yield self.aAllDirs[self.iCurrentYieldIndex]
    else:
      for sDir in self.aAllDirs:
        yield sDir

  # We could also do
  #
  #   for sFile in glob.glob(self.sInexDir/*/*.xml.txt)
  #     ...
  #
  # in one go, but that will give a VERY long list
  def __iter__(self):
    self.aAllDirs = glob.glob(os.path.join(self.sInexDir, self.sGlobPattern))
    # We always shuffle
    np.random.shuffle(self.aAllDirs)

    for sDir in self.nextDir():
      if self.bVerbose:
        print "[%s]: DIR: %s" % (self.sName, sDir)

      if os.path.isdir(sDir): # Just to be sure
        aAllFiles = glob.glob(os.path.join(sDir, "*.xml.txt"))
        # We always shuffle
        np.random.shuffle(aAllFiles)
                
        for sFile in aAllFiles:
          if self.bVerbose:
            print "[%s] FILE: %s" % (self.sName, sFile)
            
          self.sCurrentFile = sFile
          self.sPreviousSentence = None
          self.sPreviousPreviousSentence = None
          self.sPreviousParagraph = None

          fhFile = codecs.open(sFile, mode="r", encoding="utf8")
          aAllLines = [x for x in fhFile]
          fhFile.close()

          # We cannot shuffle in the non-random case, as we want consecutive
          # sentences, grouped by paragraph 
          if self.bRandom:
            np.random.shuffle(aAllLines)
          
          for sLine in aAllLines:
            if self.bVerbose:
              print "[%s] LINE: '%s'" % (self.sName, sLine.strip())

            ### Dit moet weg als het pre-processen nog een keer is gedaan
            aLine = sLine.strip().split("\t")
            sParagraph = aLine[0]
            sSentence = ' '.join(aLine[1:])
            if len(sSentence) == 0: ### THIS IS AN ERROR ACTUALLY
              print >>sys.stderr, \
                "[%s] ERROR: error in file %s" % (self.sName, sFile)
              print >>sys.stderr, \
                "[%s] ERROR: can't parse line '%s'" % (self.sName, sLine)
              continue
            ###

            # Here is the lowercasing. To be entirely safe/correct we only
            # lowercase the sentence (rather than the entire line)
            if not self.bDontLowercase:
              sSentence = sSentence.lower()

            if self.bVerbose:
              print "[%s] SENTENCE: '%s'" % (self.sName, sSentence)

            if self.bRandom or self.bSingleSentence:
              yield sSentence
            else:
              if self.sPreviousPreviousSentence is None:
                self.sPreviousPreviousSentence = sSentence
                self.sPreviousParagraph = sParagraph
              elif self.sPreviousSentence is None:
                if self.sPreviousParagraph == sParagraph:
                  self.sPreviousSentence = sSentence
                else: ## New paragraph, start again
                  self.sPreviousPreviousSentence = None
                  self.sPreviousParagraph = sParagraph
              else: ## Both previous and previous-previous sentence were seen
                if self.sPreviousParagraph == sParagraph:
                  yield (self.sPreviousSentence,         ## sentence n
                         self.sPreviousPreviousSentence, ## sentence n-1
                         sSentence)                      ## sentence n+1
                  self.sPreviousPreviousSentence = self.sPreviousSentence
                  self.sPreviousSentence = sSentence
                else: ## Different paragraph, start again
                  self.sPreviousParagraph = sParagraph
                  self.sPreviousPreviousSentence = sSentence
                  self.sPreviousSentence = None

# You can use this here below to test something
if __name__ == "__main__":
  import argparse
  oArgsParser = argparse.ArgumentParser(description='Inex utils')
  oArgsParser.add_argument('INEX_PRE_PROCESSED_DIRECTORY')
  oArgsParser.add_argument('-dont_lowercase', dest='bDontLowercase',
                           help="By default, all input text is lowercased. Use this option to prevent this.",
                           action='store_true')
  oArgsParser.add_argument("-d", dest="bDebug", action="store_true")
  oArgsParser.add_argument("-v", dest="bVerbose", action="store_true")
  oArgs = oArgsParser.parse_args()

  oSingleSentenceIterator = InexIterator(oArgs.INEX_PRE_PROCESSED_DIRECTORY,
                                         bSingleSentence=True,
                                         bDontLowercase=oArgs.bDontLowercase,
                                         sName='singleSentence',
                                         bVerbose=oArgs.bVerbose)

  i = 0
  for s in oSingleSentenceIterator:
    print s
    i += 1
    if i == 10:
      exit(1)
