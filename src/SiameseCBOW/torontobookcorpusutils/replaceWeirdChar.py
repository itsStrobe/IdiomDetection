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

if __name__ == "__main__":
  import argparse
  oArgsParser = argparse.ArgumentParser(description='Get rid of some weird characters which are irrelevant, but that do mess things up.')
  oArgsParser.add_argument("TORONTO_BOOK_CORPUS_FILE")
  oArgs = oArgsParser.parse_args()
  
  fhFile = codecs.open(oArgs.TORONTO_BOOK_CORPUS_FILE,
                       mode="r", encoding="utf8")
  sFile = fhFile.read()
  fhFile.close()

  # The next two characters cause newlines to be entered
  sFile = sFile.replace(u"\x1c", '')
  sFile = sFile.replace(u"\x1d", '')
  # Single quotes
  sFile = sFile.replace(u"\x19", " '")
  sFile = sFile.replace(u"\x18", "' ")

  sys.stdout.write(sFile)
