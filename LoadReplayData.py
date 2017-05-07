# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 06:23:13 2017

@author: ty
"""

import datetime
import os
import pandas
import sys

import h5py
from sklearn.model_selection import train_test_split
import numpy as np
import math
import json
import time
import traceback
import game as generals_game
import random
import generals_map
from bot_TNT import MAP_CHANNELS, update_state, generate_blank_state, pad, pretty_print
from game import MAX_MAP_WIDTH, ORIGINAL_MAP_WIDTH, NUM_DIRECTIONS, NORTH, EAST, SOUTH, WEST


import multiprocessing

MATCH_ID_REQUEST = 'http://halite.io/api/web/game?userID={}&limit={}'
REPLAY_REQUEST = 'https://s3.amazonaws.com/halitereplaybucket/{}'

MIN_REPLAY_STARS = 90
test = 0


def move_to_direction(game, move):
    start_y, start_x = game.index_to_coordinates(move['start'])
    end_y, end_x = game.index_to_coordinates(move['end'])
    return coordinates_to_direction(start_y, start_x, end_y, end_x)

def coordinates_to_direction(start_y, start_x, end_y, end_x):
    direction = NORTH
    
    if end_x - start_x == 1:
        direction = EAST
    elif end_x - start_x == -1:
        direction = WEST
    elif end_y - start_y == 1:
        direction = SOUTH
    
    return direction

def generate_target_move(game, move):
    is50 = int(move['is50'])
    start_y, start_x = game.index_to_coordinates(move['start'])
    end_y, end_x = game.index_to_coordinates(move['end'])
    return generate_target(game, start_y, start_x, end_y, end_x)
    
def generate_target(game, start_y, start_x, end_y, end_x):
    direction = NORTH
    
    if end_x - start_x == 1:
        direction = EAST
    elif end_x - start_x == -1:
        direction = WEST
    elif end_y - start_y == 1:
        direction = SOUTH
        
    pad_x = math.ceil(float(ORIGINAL_MAP_WIDTH - game.gmap.width) / 2.0)
    pad_y = math.ceil(float(ORIGINAL_MAP_WIDTH - game.gmap.height) / 2.0)
    return np.array([start_y + pad_y, start_x + pad_x, direction, game.gmap.height, game.gmap.width, 0, game.turn])

def generate_target_tensors(x, y, direction):
    tile_choice = np.zeros((ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH))
    direction_target = np.zeros((ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, NUM_DIRECTIONS))
    
    tile_choice[int(y), int(x)] = 1
    direction_target[int(y), int(x), int(direction)] = 1
    
    return tile_choice.flatten(), direction_target.flatten()

def load_replay(replayFolder, replay_name):
    return json.load(open('{}/{}'.format(replayFolder,replay_name)))

def load_replays(threadId, replayFolder, replayNames, file_name, lock, validation_ratio=0.1):
    print("Initializing thread {} for loading {} files!".format(threadId, len(replayNames)))
    start_time = time.time()
    
    for index, replay_name in replayNames:
        try:
            print('Loading {} on thread {} ({}/{}) with utilization in {} seconds'.format(replay_name, 
                  threadId, index, len(replayNames), time.time() - start_time))
            
            # Load replay JSON file
            replay = load_replay(replayFolder, replay_name)
            
            # Load relevant details from replay JSON file
            map_width = replay['mapWidth']
            map_height = replay['mapHeight']
            cities = replay['cities']
            cityArmies = replay['cityArmies']
            generals = replay['generals']
            mountains = replay['mountains']
            moves= replay['moves']
            afks = replay['afks']
            version = replay['version']
            players = replay['usernames']
            player_count = len(players)
            
            # Skip games that are not 1v1
            if player_count > 2:
                print("Skipping non-1v1 game...")
                continue
            
            # Skip games that does not contain at least one player of atleast MIN_REPLAY_STARS
            if max(replay['stars']) < MIN_REPLAY_STARS:
                print("Skipping game because stars are too low...")
                continue
            
            # Skip games with a player that quits
            if len(afks) > 0:
                print("Skipping game with AFK player in...")
                continue
            
            # Initialize a Game object with starting state and players
            game = generals_game.Game.from_replay(replay, version)
            game_states = [generate_blank_state(), generate_blank_state()]
            oracle_states = [generate_blank_state(), generate_blank_state()]
            replay_inputs = [[], []]
            replay_targets = [[], []]
            last_moves = [None, None]
            
            moves_count = len(moves)
            move_index = 0
            print("Beginning simulation...")
            while not game.is_over():
                if move_index >= moves_count:
                    break
                # Generate the current game state from the perspective of player with target_id index
                target_moves = [None, None]
                map_states = [game.generate_state(0), game.generate_state(1)]
                map_oracle_states = [game.generate_state(0, True), game.generate_state(1, True)]
                
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
                
                # Kill and remove AFK players from the game simulation
                if len(afks) > 0 and afks[0]['turn'] == game.turn:
                    game.kill_player(afks[0]['index'])
                
                # Add the state to training data if warranted
                for i in range(2):
                    enemy = 0 if i == 1 else 1
                    
                    target_move = target_moves[i]
                    
                    # Initialize and update oracle state
                    oracle_tiles, oracle_armies, oracle_cities, oracle_generals = map_oracle_states[i]
                    oracle_tiles = oracle_tiles.reshape(map_height, map_width)
                    oracle_armies = oracle_armies.reshape(map_height, map_width)
                    enemy_stats = (np.sum(oracle_armies[oracle_tiles == enemy]), np.sum(oracle_tiles == enemy))
                    player_stats = (np.sum(oracle_armies[oracle_tiles == i]), np.sum(oracle_tiles == i))
                    #prev_state = np.copy(game_states[i])
                    oracle_states[i] = update_state(oracle_states[i], game.turn, oracle_tiles, oracle_armies, oracle_cities, oracle_generals, i, enemy, player_stats, enemy_stats, last_moves[i])
                    current_oracle_state = np.copy(oracle_states[i])
                    
                    # Initialize and update regular game state
                    tiles, armies, cities, generals = map_states[i]
                    tiles = tiles.reshape(map_height, map_width)
                    armies = armies.reshape(map_height, map_width)
                    prev_state = np.copy(game_states[i])
                    game_states[i] = update_state(game_states[i], game.turn, tiles, armies, cities, generals, i, enemy, player_stats, enemy_stats, last_moves[i])
                    current_state = np.copy(game_states[i])
                    
                    # Skip turns that don't have a move or are randomly filtered out
                    if target_move is None:
                        continue
                    
                    # Generate the memory efficient position and direction targets
                    target = generate_target_move(game, target_move)
                    
                    # Store the last move made by this player
                    x, y, direction, height, width, winner, turn = target
                    last_moves[i] = generate_target_tensors(x, y, direction)
                    
                    # Add the oracle target
                    oracle_target = current_oracle_state[:, :, (0,1,2,3,4,5,6,10)].flatten()
                    final_target = np.concatenate((target, oracle_target), axis=0)
                    
                    # Add the final state input and target to the lists
                    replay_inputs[i].append(current_state)
                    replay_targets[i].append(np.copy(final_target))
                
                # Update the game and proceed to the next turn
                game.update()
            
            print("Ending simulation with winner {} after {} turns...".format(game.winner(), game.turn))
            game_winner = game.winner()
            
            # Check to make sure no failed games with no winner
            if game_winner is None:
                for i in range(25):                    
                    print('ERROR! ENDED GAME WITH NO WINNER!!!!! SKIPPING')
                continue
            
            print("Sampled ", (len(replay_inputs[0]) + len(replay_inputs[1])))
            # replay_input should be shape (N, 22, 22, 11) for N sampled states
            # replay_target should be shape (N, ((22, 22), (5), (1)))
            # Each sampled N state has a target, which is a:
            # 22x22 categorical prediction of the tile that moved
            # A 5-element vector denoting the direction of the movement (or still)
            # A 1-element binary vector denoting whether the movement was a 50% army move or not
            # A single target should be (tilePosition, moveDirection, is50Move)
            
            # Randomly determine whether this game will be validation or training
            dataset_name = "validation" if np.random.binomial(1, validation_ratio) else "training"
            
            # Add the sample states and targets to the thread-safe queue
            for player_id in [0, 1]:
                if (replay['stars'][player_id] < MIN_REPLAY_STARS):
                    continue
                target_length = len(replay_targets[player_id][0])
                
                inputs = np.concatenate(replay_inputs[player_id],axis=0).reshape(-1, ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, MAP_CHANNELS).astype(np.float32)
                targets = np.concatenate(replay_targets[player_id],axis=0).reshape(-1, target_length).astype(np.float32)
                
                # Set whether the player won or not in their targets
                targets[:,5] = 1 if player_id == game_winner else -1
                # Set how many turns were remaining in the game from this move onward
                targets[:, 6] = game.turn - targets[:, 6] - 1
                
                replay_inputs[player_id] = inputs
                replay_targets[player_id] = targets
                
                # Add the collected sample frames to our disk collection
                lock.acquire()
                #data_input.put(replay_inputs[player_id].astype(np.float32))
                #data_target.put(replay_targets[player_id].astype(np.float32))
                add_to_dataset(file_name, dataset_name, replay_inputs[player_id], replay_targets[player_id])
                lock.release()
        except:
            e = sys.exc_info()[0]
            print(e)
            print(traceback.format_exc())
            print("----------------------ERROR DURING LOAD! Skipping...")
            pass
    print("Thread {} is finished loading!".format(threadId))
    return 0

def fetch_replay_names(replayFolder, gamesToFetch, required_players=None):
    replayDirectory = os.listdir(replayFolder)
    random.shuffle(list(replayDirectory))
    replayNames = []
    for replay in replayDirectory:
        if len(replayNames) > gamesToFetch*2:
            break
        if replay[-10:] == '.gioreplay':
            if required_players is not None:
                #print("Load ", replay, " out of ", len(replayNames), " with a total of ", len(replayDirectory))
                replay_file = load_replay(replayFolder, replay)
                if len(replay_file['usernames']) != required_players:
                    continue
                
            replayNames.append(replay)
    replayIndices = np.random.choice(np.arange(len(replayNames)), size=min(len(replayNames), gamesToFetch), replace=False)
    replayNames = np.array(replayNames)[replayIndices]
    
    return replayNames

# Example usage: Sample 64 frames randomly from the training dataset of the FlobotFrames.h5 file
# X, y = sample_dataset("FlobotFrames", "training", 64)
def sample_dataset(file_name, dataset_name="training", sample_size = 64, sample_indices=None, data_folder='./data'):
    file_name = '{}/{}.h5'.format(data_folder, file_name)
    
    with h5py.File(file_name, 'a') as h5f:
        X_data = h5f["{}_input".format(dataset_name)]
        y_data = h5f["{}_target".format(dataset_name)]        
        
        if sample_indices is None:
            n = X_data.shape[0]
            sample_size = min(sample_size, n)
            sample_indices = np.random.choice(np.arange(n), size=sample_size, replace=False)
        
        X, y = X_data[sample_indices], y_data[sample_indices]
    
    return np.copy(X), np.copy(y)

def _dataset_exists(file_path, dataset_name, data_folder='./data'):
    if not os.path.isfile(file_path):
        return False
    
    with h5py.File(file_path, 'a') as h5f:
        exists = (dataset_name in h5f)
    
    return exists
# Example: Add some newly collected training input and target frames to dataset
# add_to_dataset("FlobotFrames", "training", input_samples, target_samples)
def add_to_dataset(file_name, dataset_name, X, y, data_folder='./data'):
    file_path = '{}/{}.h5'.format(data_folder, file_name)
    #print("Adding to file path ", file_path, " with dataset name", dataset_name)
    
    if not _dataset_exists(file_path, "{}_input".format(dataset_name), data_folder):
        _initialize_dataset(file_path, "{}_target".format(dataset_name), y, data_folder)
        _initialize_dataset(file_path, "{}_input".format(dataset_name), X, data_folder)
    else:
        _add_samples_to_dataset(file_path, "{}_input".format(dataset_name), X, data_folder)
        _add_samples_to_dataset(file_path, "{}_target".format(dataset_name), y, data_folder)

def get_dataset_info(file_name, dataset_name, data_folder='./data'):
    file_path = '{}/{}.h5'.format(data_folder, file_name)
    with h5py.File(file_path, 'a') as h5f:
        #print("Fetching info for ", file_name, " ", file_path, dataset_name)
        X_shape = h5f["{}_input".format(dataset_name)].shape
        y_shape = h5f["{}_target".format(dataset_name)].shape
    
    return X_shape, y_shape

def _initialize_dataset(file_path, dataset_name, data, data_folder='./data'):
    with h5py.File(file_path, 'a') as h5f:
        max_dataset_shape = list(data.shape)
        max_dataset_shape[0] = None
        max_dataset_shape = tuple(max_dataset_shape)
        dataset = h5f.create_dataset(dataset_name, data.shape, maxshape=max_dataset_shape)
        dataset[:] = np.copy(data)
        #print("initializing! ", max_dataset_shape, data.shape, dataset_name, h5f[dataset_name].shape)

def _add_samples_to_dataset(file_path, dataset_name, data, data_folder='./data'):
    with h5py.File(file_path, 'a') as h5f:
        dataset = h5f[dataset_name]
        n = dataset.shape[0]
        new_n = n + data.shape[0]
        dataset.resize(new_n, axis=0)
        dataset[n:] = data
        #print("Old n ", n, " versus new n", new_n)

def load_all_replays(file_name, replayFolder, gamesToLoad, threadCount=8):   
    manager = multiprocessing.Manager()
    THREADCOUNT = threadCount
    replaySubsets = [[] for thread in range(THREADCOUNT)]
    replayNames = fetch_replay_names(replayFolder, gamesToLoad, 2)
    replay_count = len(replayNames)
    for index, replay_name in enumerate(replayNames):
        replaySubsets[index % THREADCOUNT].append((index, replay_name))
        
    print("Generating training data...")
    start_time = time.time()
    threads = []
    lock = multiprocessing.Lock()
    for threadId in range(THREADCOUNT):
        threadArgs = (threadId, replayFolder, replaySubsets[threadId], file_name, lock, 0.1)
        thread = multiprocessing.Process(target=load_replays, args = threadArgs) 
        thread.daemon = True
        threads.append(thread)
        thread.start()
    
    for threadId in range(THREADCOUNT):
        print("Joining on replay loading thread {}".format(threadId))
        threads[threadId].join()
    
    load_duration = time.time() - start_time
    print("Finished loading {} games in {} seconds!".format(replay_count, load_duration))
    training_input_shape, training_target_shape = get_dataset_info(file_name, "training")
    validation_input_shape, validation_target_shape = get_dataset_info(file_name, "validation")
    print("Training data shapes: ", training_input_shape, training_target_shape)
    print("Validation data shapes: ", validation_input_shape, validation_target_shape)

if __name__ == "__main__":
    if len(sys.argv) >= 5:
        USER_ID = int(sys.argv[1])
        REPLAY_FOLDER = sys.argv[2]
        REPLAY_LIMIT = int(sys.argv[3])
        THREAD_COUNT = int(sys.argv[4])
    X, y = load_all_replays("./replays", 5, 1)
    print(X.shape)
    print(y.shape)
