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

import siamese_cbowUtils as scbowUtils
import os

if __name__ == "__main__":
  ''' This is a very simple helper script that can be called from a bash script
      to get an output file name for a call to siamese-cbow.
  '''
  oArgs = scbowUtils.parseArguments()

  # The second two options are guesses. It might be that the vocabulary is 
  # in fact smaller than iMaxNrOfVocabWords.
  sOutputFile = scbowUtils_el.makeOutputFileName(oArgs,
                                                 oArgs.iMaxNrOfVocabWords,
                                                 oArgs.iEmbeddingSize)

  print os.path.basename(sOutputFile.replace(".pickle", ''))
