#!/bin/bash

# Initialize the parameters of this bot operation
BOT_TYPE=$1
MODEL_NAME=$2

# Repeatedly run the bot against live servers in matchmaking
GAMES_PLAYED=0
while true
do
  echo "Starting game ${GAMES_PLAYED} with bot '${BOT_TYPE}' and loaded model '${MODEL_NAME}'..."
  python3 ./$1.py $2
  echo "Finished game ${GAMES_PLAYED}!"
  GAMES_PLAYED=$(expr $GAMES_PLAYED + 1)
done