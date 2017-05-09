# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 18:19:36 2017

@author: MrTwiggy
"""

import sys
from LoadReplayData import load_all_replays, copy_dataset


# --------------------- Main Logic -----------------------
# Usage: Fetch the replays from a specified folder and generate training data from the games, writing it to disk as validation and training splits.
# Example: python ./GenerateData.py ./replays/Spraget 4 1000 SpragetFrames ./data
# Will fetch 1000 replays from ./replays/Spraget and save the frames from the game into file ./data/SpragetFrames.h5
if __name__ == "__main__":
    arg_count = len(sys.argv) - 1
    
    REPLAY_FOLDER = sys.argv[1] if arg_count >= 1 else "./replays"
    THREAD_COUNT = int(sys.argv[2]) if arg_count >= 2 else 4
    GAMES_TO_LOAD = int(sys.argv[3]) if arg_count >= 3 else 100
    DATA_FILE_NAME = sys.argv[4] if arg_count >= 4 else "default-data"
    DATA_FOLDER = sys.argv[5] if arg_count >= 5 else "./data"
    
    temp_data_name = "{}-temp".format(DATA_FILE_NAME)
    
    load_all_replays(temp_data_name, REPLAY_FOLDER, GAMES_TO_LOAD, THREAD_COUNT)
    copy_dataset(temp_data_name, DATA_FILE_NAME)