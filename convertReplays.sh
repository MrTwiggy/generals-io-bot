#!/bin/bash

# Grab all files in the specified directory in command line arguments
FILES=$1/*.gior

# Convert each file from its .gior format to .gioreplay format.
for f in $FILES
do
  CONVERTED_NAME="${f}eplay"
  if [ ! -f ${CONVERTED_NAME} ]; then
    echo "Converting $f file into .gioreplay format..."
    node ./generals.io-Replay-Utils/converter.js $f
  fi
done