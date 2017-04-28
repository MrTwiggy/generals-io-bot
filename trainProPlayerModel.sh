#!/bin/bash

# Fetch the replays from pro players first
mkdir ./replays/ProPlayers
python3 ./fetch_replays.py ./replays/ProPlayers 4 Spraget 1000000 
python3 ./fetch_replays.py ./replays/ProPlayers 4 sub 1000000 
python3 ./fetch_replays.py ./replays/ProPlayers 4 bird 1000000 
python3 ./fetch_replays.py ./replays/ProPlayers 4 Firefly 1000000 
python3 ./fetch_replays.py ./replays/ProPlayers 4 Ginger 1000000 
python3 ./fetch_replays.py ./replays/ProPlayers 4 eemax 1000000 
python3 ./fetch_replays.py ./replays/ProPlayers 4 birdd 1000000
python3 ./fetch_replays.py ./replays/ProPlayers 4 0xGG 1000000
python3 ./fetch_replays.py ./replays/ProPlayers 4 2FAST4U 1000000

# Convert downloaded .gior files into plain-text JSON .gioreplay format
sudo sh convertReplays.sh ./replays/ProPlayers

# Generate replay frames from the downloaded replays
sudo python3 ./GenerateData.py ./replays/ProPlayers 4 1000000 ProPlayerFrames-v1 ./data

# Train model on all the generated replay data
sudo python3 ./train_imitation.py ProPlayerModel1 ProPlayerFrames-v1