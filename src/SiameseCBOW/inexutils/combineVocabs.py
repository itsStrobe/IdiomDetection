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

import sys
import glob
import os
import collections
import codecs

sys.path.append("../siamese-skipgram")
import vocabUtils

def readCountWord(sFile, sCols=None):
  fhFile = codecs.open(sFile, mode='r', encoding='utf8')

  dVocab = {}

  iLineNr = 0
  for sLine in fhFile:
    iLineNr += 1
    try:
      sWord, sFile = None, None
      if sCols == "wordfreq":
        sWord, sFreq = sLine.strip().split("\t")
      else: # "freqword"
        sFreq, sWord = sLine.strip().split("\t")

      if sWord in dVocab:
        print >>sys.stderr, \
            "[WARNING]: word '%s' is already in" % sWord
      else:
        dVocab[sWord] = int(sFreq)
    except ValueError:
      sLine = sLine[:-1] if sLine.endswith("\n") else sLine
      print >>sys.stderr, \
          "[WARNING]: error in line %d: '%s'" % (iLineNr, sLine)
   
  fhFile.close()

  return dVocab

if __name__ == "__main__":
  import argparse
  oArgsParser = argparse.ArgumentParser(description='Bla bla')
  oArgsParser.add_argument('VOCAB_DIRECTORY')
  oArgsParser.add_argument('OUTPUT_FILE')  
  oArgsParser.add_argument("-cols", metavar="COLS",
                           help="Specifies column order. freqword, or wordfreq. Default: 'freqword'",
                           dest="sCols", action="store",
                           choices=["freqword", "wordfreq"],
                           default='freqword')
  oArgsParser.add_argument("-d", dest="bDebug", action="store_true")
  oArgsParser.add_argument("-v", dest="bVerbose", action="store_true")
  oArgs = oArgsParser.parse_args()

  oCounter = collections.Counter()

  for sVocabFile in \
        glob.glob(os.path.join(oArgs.VOCAB_DIRECTORY, "*.vocab.txt")):
    if oArgs.bVerbose:
      print "Reading %s" % sVocabFile
    dVocab = readCountWord(sVocabFile, sCols=oArgs.sCols)
    
    oCounter.update(dVocab)

  if oArgs.bVerbose:
    print "Storing to %s" % oArgs.OUTPUT_FILE
  fhOut = codecs.open(oArgs.OUTPUT_FILE, mode='w', encoding='utf8')
  # Output in sorted order
  for sKey, iFreq in oCounter.most_common(None):
    # NOTE: we now write the other way around, so frequency first
    fhOut.write("%d %s\n" % (iFreq, sKey))
  fhOut.close()

