#!/bin/bash

scp -r ./src/Word2Vec/models/     jz18436@ceres.essex.ac.uk:./IdiomDetection/src/Word2Vec/
scp -r ./src/SiameseCBOW/models/  jz18436@ceres.essex.ac.uk:./IdiomDetection/src/SiameseCBOW/
scp -r ./src/SkipThoughts/models/ jz18436@ceres.essex.ac.uk:./IdiomDetection/src/SkipThoughts/
scp -r ./src/ELMo/models/         jz18436@ceres.essex.ac.uk:./IdiomDetection/src/ELMo/
