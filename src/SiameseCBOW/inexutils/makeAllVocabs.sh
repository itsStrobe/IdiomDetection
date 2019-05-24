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

if [ -z "$3" ]; then
    echo
    echo " USAGE: $0 NR_OF_PROCESSES PRE_PROCESSED_INEX_DIRECTORY OUTPUT_DIR"
    echo
    exit
fi

NR_OF_PROCESSES=$1
PRE_PROCESSED_INEX_DIRECTORY=$2
OUTPUT_DIR=$3

for I in $(seq 0 9); do
    for J in $(seq 0 9); do
        for K in $(seq 0 9); do
            OUTPUT_FILE=$OUTPUT_DIR/$I$J$K.vocab.txt

            echo "Doing $I$J$K -> $OUTPUT_FILE"
            python countWords.py -glob_pattern $I$J$K $PRE_PROCESSED_INEX_DIRECTORY $OUTPUT_FILE &
            
            NUM_CHILDREN=$(pgrep -P $PARENT_ID | wc -l)
            while [ $NUM_CHILDREN -ge $NR_OF_PROCESSES ]; do
                sleep 0.1
                NUM_CHILDREN=$(($(pgrep -P $PARENT_ID | wc -l) - 1))
            done
        done
    done
done

wait
