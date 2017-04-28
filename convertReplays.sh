#!/bin/bash

# Grab all files in the current working directory
FILES=$PWD/*

# Convert each file from its .gior format to .gioreplay format.
for f in $FILES
do
  echo "Converting $f file into .gioreplay format..."
  node ./generals.io-Replay-Utils/converter.js $f
done