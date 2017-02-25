# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 06:23:13 2017

@author: ty
"""

import datetime
import os
import sys

import numpy as np
import math
import json
import time
from bot_TNT import MAX_MAP_WIDTH, MAP_CHANNELS, update_state

import multiprocessing

MATCH_ID_REQUEST = 'http://halite.io/api/web/game?userID={}&limit={}'
REPLAY_REQUEST = 'https://s3.amazonaws.com/halitereplaybucket/{}'

if len(sys.argv) >= 5:
    USER_ID = int(sys.argv[1])
    REPLAY_FOLDER = sys.argv[2]
    REPLAY_LIMIT = int(sys.argv[3])
    THREAD_COUNT = int(sys.argv[4])

def load_replays(threadId, replayFolder, replayNames, data_input, data_target, lock):
    print("Initializing thread {} for loading {} files!".format(threadId, len(replayNames)))
    for index, replay_name in replayNames:
        try:
            print('Loading {} on thread {} ({}/{}) with utilization of ({} / {})'.format(replay_name, threadId, index, len(replayNames), total_lock, total_calc))
            replay = json.load(open('{}/{}'.format(replayFolder,replay_name)))
            
            map_width = replay['mapWidth']
            map_height = replay['mapHeight']
            cities = replay['cities']
            cityArmies = replay['citiyArmies']
            generals = replay['generals']
            mountains = replay['mountains']
            moves= replay['moves']
            afks = replay['afks']
            players = replay['usernames']
            player_count = len(players)
            
            
            # Initialize a Game object with starting state and players
            game = initialize_game(map_width, map_height, players, cities, cityArmies, generals, mountains)
            
            # Play through the game, and sample random states and actions from target_id player
            target_id = 0
            
            replay_input = []
            replay_target = []
            
            moves_count = len(moves)
            move_index = 0
            while not game.is_done():
                # Generate the current game state from the perspective of player with target_id index
                game_state = game.generate_state(target_id)
                target_move = None
                
                # Submit all moves and simulate the game state from replay
                while move_index < moves_count and moves[move_index].turn <= game.turn:
                    move = moves[move_index]
                    move_index += 1
                    game.handle_move(move)
                    if (move['index'] == target_id):
                        target_move = move
                
                # Add the state to training data if warranted
                if target_move is not None:
                    replay_input.append(game_state)
                    replay_target.append(generate_target(target_move))
                
                # Update the game and proceed to the next turn
                game.update()
            
            # replay_input should be shape (N, 22, 22, 11) for N sampled states
            # replay_target should be shape (N, ((22, 22), (5), (1)))
            # Each sampled N state has a target, which is a:
            # 22x22 categorical prediction of the tile that moved
            # A 5-element vector denoting the direction of the movement (or still)
            # A 1-element binary vector denoting whether the movement was a 50% army move or not
            # A single target should be (tilePosition, moveDirection, is50Move)
            
            # Reduce the total training data set from this replay by sampling select states            
            
            # Add the sample states and targets to the thread-safe queue
            lock.acquire()
            data_input.put(replay_input.astype(np.float32))
            data_target.put(replay_target.astype(np.float32))
            lock.release()
        except:
            e = sys.exc_info()[0]
            print(e)
            print("----------------------ERROR DURING LOAD! Skipping...")
            pass
    print("Thread {} is finished loading!".format(threadId))
    return 0


    
def queue_to_list(queue):
    result = []
    while queue.qsize() != 0:
        result.append(queue.get())
    return result

def load_all_replays(replayFolder, gamesToLoad, threadCount=8):   
    manager = multiprocessing.Manager()
    data_input = manager.Queue()
    data_target = manager.Queue()
    THREADCOUNT = threadCount
    replaySubsets = [[] for thread in range(THREADCOUNT)]
    replayDirectory = os.listdir(replayFolder)
    replayNames = []
    for replay in replayDirectory:
        if replay[-10:] == '.gioreplay':
            replayNames.append(replay)
    replayIndices = np.random.choice(np.arange(len(replayNames)), size=min(len(replayNames), gamesToLoad), replace=False)
    replayNames = np.array(replayNames)[replayIndices]
    for index, replay_name in enumerate(replayNames):
        replaySubsets[index % THREADCOUNT].append((index, replay_name))
        
    print("Generating training data...")
    start_time = time.time()
    threads = []
    lock = multiprocessing.Lock()
    for threadId in range(THREADCOUNT):
        threadArgs = (threadId, replayFolder, replaySubsets[threadId], data_input, data_target, lock,)
        thread = multiprocessing.Process(target=load_replays, args = threadArgs) 
        thread.daemon = True
        threads.append(thread)
        thread.start()
    
    for threadId in range(THREADCOUNT):
        print("Joining on thread {}".format(threadId))
        threads[threadId].join()
    
    load_duration = time.time() - start_time
    print("Finished loading {} games in {} seconds!".format(data_input.qsize(), load_duration))
    
    training_input = queue_to_list(data_input)
    training_target = queue_to_list(data_target)
    
    print("Finished loading {} games in {} seconds!".format(len(training_input), load_duration))
    
    training_input = np.concatenate(training_input,axis=0)
    training_target = np.concatenate(training_target,axis=0)
    
    print("Finishing loading data, now beginning to save to disk...")
    print("Finished loading and saving data!")
    return training_input, training_target
