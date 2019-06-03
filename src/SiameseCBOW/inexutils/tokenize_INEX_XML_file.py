# -*- coding: utf-8 -*-

import nltk
import sys
from BeautifulSoup import BeautifulSoup
import codecs
import re

sys.path.append('../tokenizationutils')
import tokenizer

def process_line(fhOut, sLine, iNewParagraphMarker,
                 oPunktSentenceTokenizer,
                 bLowerCase=False, bDebug=False):
  if bDebug:
    import pdb
    pdb.set_trace()

  # In paragraphh mode, only text in <p> tags is considered
  if (iNewParagraphMarker is not None) and \
        (not (sLine.startswith("<p ") or sLine.startswith("<p>"))):
    return iNewParagraphMarker

  if sLine.startswith("<?xml") or sLine.startswith("</xml>") or \
        sLine.startswith("<page>") or \
        sLine.startswith("</page>") or \
        sLine.startswith("<a>") or sLine.startswith("</a>") or  \
        sLine.startswith("<ID>"):
    return iNewParagraphMarker

  # Sometimes (but not always) the ampersands in HTML entities are encoded
  # So like &amp;nbsp; or &amp;#9775;
  sLine = sLine.replace("&amp;", "&")
  oSoup = BeautifulSoup(sLine, 
                        convertEntities=BeautifulSoup.HTML_ENTITIES).findAll(text=True)

  sText = ''.join([sLine for sLine in oSoup])

  # TABs may occur?!?
  sText = sText.replace("\t", ' ')
  # And other strange white-spacy characters
  sText = sText.replace(u" ", ' ')
  sText = sText.replace(u"", ' ')

  # Now that we might have introduced some subsequent spaces, delete (this
  # makes processing easier down-stream from here)
  sText = re.sub("\s+", " ", sText)

  sPrefix = "%d\t" % iNewParagraphMarker if (iNewParagraphMarker is not None)\
      else ''

  for sSentence in oPunktSentenceTokenizer.tokenize(sText.strip()):
    # Remove laiding and trailing non-word characters for every word
    sSentence = tokenizer.removeNonTokenChars(sSentence).strip()

    if len(sSentence): # If there is something left to print
      if bLowerCase:
        fhOut.write("%s%s\n" % (sPrefix, sSentence.lower()))
      else:
        fhOut.write("%s%s\n" % (sPrefix, sSentence))
  
  # If iNewParagraphMarker = 0, abs(iNewParagraphMarker - 1) => 1
  # If iNewParagraphMarker = 1, abs(iNewParagraphMarker - 0) => 0
  return None if iNewParagraphMarker is None else abs(iNewParagraphMarker - 1)

if __name__ == "__main__":
  import argparse
  oArgsParser = argparse.ArgumentParser(description='Tokenize a single INEX (wikipedia) XML file. Output will be one sentence per line.')
  oArgsParser.add_argument('INEX_XML_FILE')
  oArgsParser.add_argument('-paragraph_mode',
                           dest="bParagraphMode",
                           help="In this mode, only text in <p>tags is considered. Also, preceding every sentence, a paragraph token is printed that discriminates it from the previous paragraph (alternating 0s and 1s are used). ",
                           action="store_true")
  oArgsParser.add_argument('-o', dest="sOutputFile", metavar="FILE",
                           help="Output file (if not provided, use stdout)",
                           action="store")  
  oArgsParser.add_argument('-l', dest="bLowerCase",
                           help="Lowercase everything", action="store_true")  
  oArgsParser.add_argument('-d', help='Debugger mode',
                           dest="bDebug", action="store_true")
  oArgsParser.add_argument('-v', help="Be verbose", dest='bVerbose',
                           action='store_true')
  oArgs = oArgsParser.parse_args()

  if oArgs.bDebug:
    import pdb
    pdb.set_trace()

  fhOut = None
  if oArgs.sOutputFile is not None:
    fhOut = codecs.open(oArgs.sOutputFile, mode='w', encoding='utf8')
  else:
    fhOut = codecs.getwriter("utf-8")(sys.stdout)

  # Load pre-trained English sentence splitter
  oPunktSentenceTokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  iNrOfSentencesProcessed = 0

  iNrOfLines = 0
  try:
    fhFile = codecs.open(oArgs.INEX_XML_FILE, mode='r', encoding='utf8')
    
    iNewParagraphMarker = 0 if oArgs.bParagraphMode else None
    for sLine in fhFile:        
      iNewParagraphMarker = \
          process_line(fhOut, sLine.strip(), iNewParagraphMarker,
                       oPunktSentenceTokenizer, 
                       bLowerCase=oArgs.bLowerCase, bDebug=oArgs.bDebug)
      
      iNrOfLines += 1
      if (iNrOfLines % 10000) == 0:
        if oArgs.bVerbose:
          print >> sys.stderr, "Processed %d lines" % iNrOfLines

    fhFile.close()
  except Exception as error:
    print type(error), "in", oArgs.INEX_XML_FILE + ":", error
    import traceback
    traceback.print_exc()

  if oArgs.bVerbose:
    print >> sys.stderr, "Processed %d lines in total" % iNrOfLines
