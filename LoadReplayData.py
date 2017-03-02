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
import traceback
import game as generals_game
import generals_map
from bot_TNT import MAX_MAP_WIDTH, MAP_CHANNELS, update_state, generate_blank_state, pad

import multiprocessing

MATCH_ID_REQUEST = 'http://halite.io/api/web/game?userID={}&limit={}'
REPLAY_REQUEST = 'https://s3.amazonaws.com/halitereplaybucket/{}'

if len(sys.argv) >= 5:
    USER_ID = int(sys.argv[1])
    REPLAY_FOLDER = sys.argv[2]
    REPLAY_LIMIT = int(sys.argv[3])
    THREAD_COUNT = int(sys.argv[4])

STILL = 0
NORTH = 1
EAST = 2
SOUTH = 3
WEST = 4

def generate_target(game, move):
    tile_choice = np.zeros((game.gmap.height, game.gmap.width))
    if move is None:
        is50 = 0
        direction = STILL
    else:
        is50 = int(move['is50'])
        start_y, start_x = game.index_to_coordinates(move['start'])
        end_y, end_x = game.index_to_coordinates(move['end'])
        
        tile_choice[start_y, start_x] = 1
        
        direction = NORTH
        
        if end_x - start_x == 1:
            direction = EAST
        elif end_x - start_x == -1:
            direction = WEST
        elif end_y - start_y == 1:
            direction = SOUTH
    
    tile_choice, _, _ = pad(tile_choice, 0)
    direction_target = (np.arange(5) == direction)
    return np.concatenate((tile_choice.flatten(), direction_target.flatten(), np.array([is50])), axis=0)

def load_replays(threadId, replayFolder, replayNames, data_input, data_target, lock):
    print("Initializing thread {} for loading {} files!".format(threadId, len(replayNames)))
    for index, replay_name in replayNames:
        try:
            print('Loading {} on thread {} ({}/{}) with utilization'.format(replay_name, threadId, index, len(replayNames)))
            replay = json.load(open('{}/{}'.format(replayFolder,replay_name)))
            
            map_width = replay['mapWidth']
            map_height = replay['mapHeight']
            cities = replay['cities']
            cityArmies = replay['cityArmies']
            generals = replay['generals']
            mountains = replay['mountains']
            moves= replay['moves']
            afks = replay['afks']
            
            players = replay['usernames']
            player_count = len(players)
            
            if player_count > 2:
                print("SKipping replay...")
                print(players)
                print(afks)
                continue
            
            # Initialize a Game object with starting state and players
            game = generals_game.Game.from_replay(replay)
            game_states = [generate_blank_state(), generate_blank_state()]
            # Play through the game, and sample random states and actions from target_id player
            target_id = 0
            
            replay_inputs = [[], []]
            replay_targets = [[], []]
            
            moves_count = len(moves)
            move_index = 0
            print("Beginning simulation...")
            while not game.is_over():
                # Generate the current game state from the perspective of player with target_id index
                #game_state = game.generate_state(target_id)
                target_moves = [None, None]
                map_states = [game.generate_state(0), game.generate_state(1)]
                #print("Starting turn: {}".format(game.turn))
                # Submit all moves and simulate the game state from replay
                while move_index < moves_count and moves[move_index]['turn'] <= game.turn:
                    move = moves[move_index]
                    move_index += 1
                    player = move['index']
                    start = move['start']
                    end = move['end']
                    is50 = move['is50']
                    success = game.handle_attack(player, start, end, is50)
                    target_moves[player] = move
                
                if len(afks) > 0 and afks[0]['turn'] == game.turn:
                    game.kill_player(afks[0]['index'])
                    print("KIlling AFK player ", afks[0]['index'])
                
                if ((game.turn - 1) % 1500 == 0):
                    game.print_game()
                
                # Add the state to training data if warranted
                for i in range(2):
                    enemy = 0 if i == 1 else 1
                    
                    target_move = target_moves[i]
                    
                    if target_move == None and np.random.binomial(1, 0.95):
                        continue
                    if target_move is not None and np.random.binomial(1, 0.95):
                        continue
                    tiles, armies, cities, generals = map_states[i]
                    tiles = tiles.reshape(map_height, map_width)
                    armies = armies.reshape(map_height, map_width)
                    game_states[i] = update_state(game_states[i], tiles, armies, cities, generals, i, enemy)
                    replay_inputs[i].append(game_states[i])
                    target = generate_target(game, target_move)
                    print("Target shape ", target.shape)
                    replay_targets[i].append(target)
                
                # Update the game and proceed to the next turn
                game.update()
            
            print("Ending simulation with winner {} after {} turns...".format(game.winner(), game.turn))
            game_winner = game.winner()
            print("Sampled ", len(replay_inputs[game_winner]))
            # replay_input should be shape (N, 22, 22, 11) for N sampled states
            # replay_target should be shape (N, ((22, 22), (5), (1)))
            # Each sampled N state has a target, which is a:
            # 22x22 categorical prediction of the tile that moved
            # A 5-element vector denoting the direction of the movement (or still)
            # A 1-element binary vector denoting whether the movement was a 50% army move or not
            # A single target should be (tilePosition, moveDirection, is50Move)
            
            # Reduce the total training data set from this replay by sampling select states            
            
            # Add the sample states and targets to the thread-safe queue
            target_length = MAX_MAP_WIDTH*MAX_MAP_WIDTH + 6
            replay_inputs[game_winner] = np.concatenate(replay_inputs[game_winner],axis=0).reshape(-1, 22, 22, 11)
            replay_targets[game_winner] = np.concatenate(replay_targets[game_winner],axis=0).reshape(-1, target_length)
            print("Shape: ", replay_inputs[game_winner].shape, replay_targets[game_winner].shape)
            lock.acquire()
            data_input.put(replay_inputs[game_winner].astype(np.float32))
            data_target.put(replay_targets[game_winner].astype(np.float32))
            lock.release()
        except:
            e = sys.exc_info()[0]
            print(e)
            print(traceback.format_exc())
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


if __name__ == "__main__":
    X, y = load_all_replays("./replays", 5, 1)
    print(X.shape)
    print(y.shape)