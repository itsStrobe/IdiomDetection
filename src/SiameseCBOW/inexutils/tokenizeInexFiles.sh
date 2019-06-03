#!/bin/sh

# Copyright 2016 Tom Kenter
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

PARENT_ID=$$

if [ -z "$4" ]; then
    echo
    echo " USAGE: $0 INPUT_DIR DIR_NRS OUTPUT_DIR NR_OF_PROCESSES [PARAGRAPH_MODE]"
    echo 
    echo "   E.g.: $0 /zfs/ilps-plexest/INEX/data/INEXQAcorpus2013/ 55 myDirOfInexTokenizedFiles 10"
    echo
    echo "   If you call it like this it will start 10 parallel processes that will"
    echo "   tokenize ALL INEX files in directories starting with 55 (so files in"
    echo "   /zfs/ilps-plexest/INEX/data/INEXQAcorpus2013/550,"
    echo "   /zfs/ilps-plexest/INEX/data/INEXQAcorpus2013/551, etc.)."
    echo "   If you specify '*' anything will match"
    echo 
    echo "   The result is stored in the OUTPUT_DIR in which the original directory"
    echo "   structure of the INEX directory will be built"
    echo
    echo "   Paragraph mode is induced by providing -paragraph_mode."
    echo "   For normal mode, just leave out this option."
    echo
    exit
fi

INPUT_DIR=$1
DIR_NRS=$2
OUTPUT_DIR=$3
NR_OF_PROCESSES=$4
PARAGRAPH_MODE="$5"

BASE_DIR=$(basename $INPUT_DIR)

NR_OF_FILES=0
for DIR in $(ls $INPUT_DIR | grep "^$DIR_NRS" ); do
    for XML_FILE in $(ls $INPUT_DIR/$DIR/*.xml); do
        #if [ $NR_OF_FILES -eq 1000 ]; then
        #    exit
        #fi
        
        FILE_BASE_NAME=$(basename $XML_FILE)
        OUTPUT_FILE=$OUTPUT_DIR/$DIR/$FILE_BASE_NAME.txt

        NR_OF_FILES=$((NR_OF_FILES+1))

        if [ ! -d $OUTPUT_DIR/$DIR ]; then
            echo "Making directory $OUTPUT_DIR/$DIR"
            mkdir $OUTPUT_DIR/$DIR
        fi

        echo $NR_OF_FILES" "$DIR" "$FILE_BASE_NAME.txt
        python tokenize_INEX_XML_file.py $PARAGRAPH_MODE -o $OUTPUT_FILE $XML_FILE &

        NUM_CHILDREN=$(pgrep -P $PARENT_ID | wc -l)
        while [ $NUM_CHILDREN -ge $NR_OF_PROCESSES ]; do
            sleep 0.1
            NUM_CHILDREN=$(($(pgrep -P $PARENT_ID | wc -l) - 1))
        done
    done
done

wait
