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

import codecs
import sys
import subprocess
import numpy as np
import time
import cPickle
import os

def file2positions(sFile):
  sCommand = "wc -l < %s" % sFile
  # When wc reads from stdin it just gives a number as output
  iNrOfLines = int(subprocess.check_output(sCommand, shell=True))

  npaPositions = np.empty(iNrOfLines, dtype=np.uint32)

  fhOut = codecs.open("bla.txt", mode='w', encoding='utf8')

  iSentenceIndex = 0
  iSentencePos = 0
  fhFile = codecs.open(sFile, mode='r', encoding='utf8')
  for sLine in fhFile:
    if iSentenceIndex == iNrOfLines:
      print >>sys.stderr, "[WARNING] reached the end %d. Last line was '%s'"%\
          (iSentenceIndex, sLine)
      break

    fhOut.write("%d\t%s" % (iSentenceIndex, sLine))
    npaPositions[iSentenceIndex] = iSentencePos
    iSentenceIndex += 1
    iSentencePos += len(sLine.encode("utf8"))

  fhFile.close()
  fhOut.close()

  return npaPositions

if __name__ == "__main__":
  import argparse
  oArgsParser = argparse.ArgumentParser(description='This script generates a file with character offsets of the sentences in a Toronto Book Corpus file.')
  oArgsParser.add_argument("FILE")
  oArgsParser.add_argument("OUTPUT_DIR")
  oArgsParser.add_argument("-d", dest="bDebug", action="store_true")
  oArgsParser.add_argument("-v", dest="bVerbose", action="store_true")
  oArgs = oArgsParser.parse_args()

  fStartTime = None
  if oArgs.bVerbose:
    print "Start calculating positions"
    fStartTime = time.time()

  npaSentencePositions = file2positions(oArgs.FILE)

  if oArgs.bVerbose:
    fEndTime = time.time() if oArgs.bVerbose else None
    fTotalSeconds = fEndTime - fStartTime
    iHours = int(fTotalSeconds/3600)
    iMinutes = int((fTotalSeconds % 3600) / 60)
    fSeconds = fTotalSeconds % 60

    sHours = '' if iHours == 0 else "%d hours, " % iHours \
        if iHours > 1 else "%d hour, " % iHours 
    sMinutes = "%d minutes" % iMinutes if iMinutes != 1 \
        else "%d minute" % iMinutes
    print \
        "Calculating pos took %s%s and %.2f seconds (%f seconds in total)" % \
        (sHours, sMinutes, fSeconds, fTotalSeconds)

  sOutputFile = os.path.join(oArgs.OUTPUT_DIR, os.path.basename(oArgs.FILE))
  sOutputFile.replace(".txt", '')
  sOutputFile = "%s.pickle" % sOutputFile

  if oArgs.bVerbose:
    print "Start pickling to %s" % sOutputFile
    fStartTime = time.time()

  fhOutputFile = open(sOutputFile, mode='wb')
  cPickle.dump(npaSentencePositions, fhOutputFile)
  fhOutputFile.close()

  if oArgs.bVerbose:
    fEndTime = time.time() if oArgs.bVerbose else None
    fTotalSeconds = fEndTime - fStartTime
    iHours = int(fTotalSeconds/3600)
    iMinutes = int((fTotalSeconds % 3600) / 60)
    fSeconds = fTotalSeconds % 60

    sHours = '' if iHours == 0 else "%d hours, " % iHours \
        if iHours > 1 else "%d hour, " % iHours 
    sMinutes = "%d minutes" % iMinutes if iMinutes != 1 \
        else "%d minute" % iMinutes
    print \
        "Pickling took %s%s and %.2f seconds (%f seconds in total)" % \
        (sHours, sMinutes, fSeconds, fTotalSeconds)
