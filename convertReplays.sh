#!/bin/bash

# Grab all files in the specified directory
FILES=$1/*

# Convert each file from its .gior format to .gioreplay format.
for f in $FILES
do
  echo "Converting $f file into .gioreplay format..."
  node ./generals.io-Replay-Utils/converter.js $f
done
