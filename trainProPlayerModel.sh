#!/bin/bash

# Install requirements
sudo pip3 install -r requirements.txt

# Fetch the replays from pro players first
sudo mkdir ./replays/ProPlayers
sudo python3 ./fetch_replays.py ./replays/ProPlayers 4 [Pros] 1000000 60 100 
 
# Convert downloaded .gior files into plain-text JSON .gioreplay format
sudo sh convertReplays.sh ./replays/ProPlayers
 
# Generate replay frames from the downloaded replays
sudo mkdir ./data
sudo python3 ./GenerateData.py ./replays/ProPlayers 4 1000000 ProPlayerFrames-v1 ./data
 
# Train model on all the generated replay data
python3 ./train_imitation.py ProPlayerModel1 ProPlayerFrames-v1