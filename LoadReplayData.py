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

test = 0


def move_to_direction(game, move):
    start_y, start_x = game.index_to_coordinates(move['start'])
    end_y, end_x = game.index_to_coordinates(move['end'])
    
    
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
    return np.array([start_y + pad_y, start_x + pad_x, direction, game.gmap.height, game.gmap.width])

def generate_target_tensors(x, y, direction):
    tile_choice = np.zeros((ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH))
    direction_target = np.zeros((ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, NUM_DIRECTIONS))
    
    tile_choice[int(y), int(x)] = 1
    direction_target[int(y), int(x), int(direction)] = 1
    
    return tile_choice.flatten(), direction_target.flatten()

def load_replay(replayFolder, replay_name):
    return json.load(open('{}/{}'.format(replayFolder,replay_name)))

def load_replays(threadId, replayFolder, replayNames, data_input, data_target, lock,max_width):
    print("Initializing thread {} for loading {} files!".format(threadId, len(replayNames)))
    start_time = time.time()
    player_tracker = {}
    #from train_imitation import load_model_train
    #model = load_model_train("./data", "april16-earlygame-newv1")
    for index, replay_name in replayNames:
        try:
            
            broken_replay = False
            print('Loading {} on thread {} ({}/{}) with utilization in {} seconds'.format(replay_name, 
                  threadId, index, len(replayNames), time.time() - start_time))
            
            replay = load_replay(replayFolder, replay_name)
            
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
            else:
                print(players[0], players[1])
            
            
            with lock:
                #if map_width > max_width.value:
                #    max_width.value = map_width
                #if map_height > max_width.value:
                #    max_width.value = map_height
                #if players[0] == 'Spraget' or players[1] == 'Spraget':
                print("Games found: {}".format(max_width.value))
                """for player in players:
                    if player not in player_tracker:
                        player_tracker[player] = 0
                    player_tracker[player] += 1
                continue
                    """
                if max(replay['stars']) >= 90:
                    print("grrr", max(replay['stars']))
                    z = 1
                    max_width.value += 1
                    if max_width.value >= 10000:
                        break
                else:
                    
                    print("grrr222", max(replay['stars']))
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
            corrects = [0.0, 0.0]
            total = [0.0, 0.0]
            while not game.is_over():
                if (len(afks) == 0 or afks[0]['turn'] < game.turn) and move_index >= moves_count:
                    broken_replay = True
                    break
                
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
                
                
                # Add the state to training data if warranted
                for i in range(2):
                    enemy = 0 if i == 1 else 1
                    
                    target_move = target_moves[i]
                    tiles, armies, cities, generals = map_states[i]
                    tiles = tiles.reshape(map_height, map_width)
                    armies = armies.reshape(map_height, map_width)
                    #old_state = np.copy(game_states[i])
                    game_states[i] = update_state(game_states[i], tiles, armies, cities, generals, i, enemy)
                    map_state = game_states[i]
                    if i != -1 and index == -1:
                        print("--------Fog tiles-------------")
                        pretty_print(map_state[:,:,7])
                        print("--------Discovered tiles-------------")
                        pretty_print(map_state[:,:,8])
                        print("--------Turns in fog tiles-------------", replay_name)
                        pretty_print(map_state[:,:,9])
                        print("--------Army tiles-------------", replay_name, i)
                        pretty_print(map_state[:,:,10])
                    
                    if target_move == None:# and np.random.binomial(1, 0.98):
                        continue
                    if target_move is not None and np.random.binomial(1, 0.4):
                        continue
                    if target_move is not None and game.turn > 200000:
                        break
                    
                    replay_inputs[i].append(np.copy(game_states[i]))
                    target = generate_target_move(game, target_move)
                    #print("Target shape ", target.shape)
                    replay_targets[i].append(np.copy(target))
                    """
                    tile_pos, move_direction = model.predict(old_state.reshape(-1, 31, 31, 11))
                    tile_pos = tile_pos.flatten().reshape(-1, 529)
                    move_direction = move_direction.reshape(-1, ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, 4)
                    
                    diff_y = float(ORIGINAL_MAP_WIDTH - game.gmap.height)
                    diff_x = float(ORIGINAL_MAP_WIDTH - game.gmap.width)    
                    y_pad = math.ceil(diff_y / 2.0)
                    x_pad = math.ceil(diff_x / 2.0)
                    
                    tile_index = np.argmax(tile_pos.flatten())
                    
                    total[0] += 1
                    total[1] += 1
                    start_y, start_x = game.index_to_coordinates(target_move['start'])
                    #print(move_direction.shape, tile_pos.shape)
                    target_tile_index = np.argmax(target[:529].flatten())
                    orig_y = y_pad + start_y
                    orig_x = x_pad + start_x
                    target_direction = np.argmax(target[529:].reshape(23, 23, 4)[orig_y, orig_x])# move_to_direction(game, target_move)
                    if np.argmax(move_direction[0, orig_y, orig_x]) == target_direction:
                        corrects[0] += 1
                    if tile_index == target_tile_index:
                        corrects[1] += 1
                    
                    test_target = target[529:].reshape(23, 23, 4)
                    test = np.any(test_target, axis=-1)
                    test_pos = np.argmax(test.flatten())
                    test_y, test_x = np.unravel_index(test_pos, test.shape)
                    #print("Test: ", test_y, test_x, " VS ", orig_y, orig_x)
                    
                    model_input = np.array([old_state])
                    model_target = [np.array([target[:529].flatten()]), np.array([target[529:].flatten()])]
                    #print(model_input.shape, model_target[0].shape, model_target[1].shape)
                    #a = model.evaluate(model_input, model_target)
                    #print("eval----", a)                    
                    #print("Move direction {} out of {}, tile position {} out of {} with {} vs {} and {} vs {} for player {}".format(corrects[0] / total[0], total[0],
                    #      corrects[1] / total[1], total[1], tile_index, target_tile_index, np.argmax(move_direction[orig_y, orig_x]), target_direction, i))
                    """
                
                # Update the game and proceed to the next turn
                game.update()
            
            if broken_replay:
                print("Ending simulation as it failed to produce valid results in completion!")
                continue
            print("Ending simulation with winner {} after {} turns...".format(game.winner(), game.turn))
            game_winner = game.winner()
            
            print("Sampled ", len(replay_inputs[game_winner]))
            if 'Spraget' == players[0] or 'Spraget' == players[1]:
                print("Spraget! ")
                if 'Spraget' == players[0]:
                    game_winner = 0
                else:
                    game_winner = 1
            #else:
            #    continue
            
            pro_names = ['Spraget', 'sub', 'Firefly', 'sora', '[Bot] FloBot']
            for i in [0, 1]:
                for pro_name in pro_names:
                    if players[i] == pro_name:
                        game_winner = i
                        found = True
                        
            if not found:
                continue
            # replay_input should be shape (N, 22, 22, 11) for N sampled states
            # replay_target should be shape (N, ((22, 22), (5), (1)))
            # Each sampled N state has a target, which is a:
            # 22x22 categorical prediction of the tile that moved
            # A 5-element vector denoting the direction of the movement (or still)
            # A 1-element binary vector denoting whether the movement was a 50% army move or not
            # A single target should be (tilePosition, moveDirection, is50Move)
            
            # Reduce the total training data set from this replay by sampling select states            
            
            # Add the sample states and targets to the thread-safe queue
            for game_winner in [game_winner]:
                target_length = len(replay_targets[game_winner][0])
                replay_inputs[game_winner] = np.concatenate(replay_inputs[game_winner],axis=0).reshape(-1, ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, MAP_CHANNELS)
                replay_targets[game_winner] = np.concatenate(replay_targets[game_winner],axis=0).reshape(-1, target_length)
                
                
                sample_index = len(replay_inputs[game_winner])
                model_in = replay_inputs[game_winner][:sample_index]
                model_target = [replay_targets[game_winner][:sample_index, :ORIGINAL_MAP_WIDTH**2], replay_targets[game_winner][:sample_index, ORIGINAL_MAP_WIDTH**2:]]
                print("Shape: ", model_in.shape, model_target[0].shape, model_target[1].shape)
                #a = model.evaluate(model_in, model_target)
                #print("------------EVALUTIOATIOF: ", a)                
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
    import operator
    print(sorted(player_tracker.items(), key=operator.itemgetter(1)))
    return 0

def queue_to_list(queue):
    result = []
    while queue.qsize() != 0:
        result.append(queue.get())
    return result

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

def load_data(data_name, data_folder='./data'):
    file_name = '{}/{}.h5'.format(data_folder, data_name)
    h5f = h5py.File(file_name, 'r')
    training_input = h5f['training_input'][:]
    training_target = h5f['training_target'][:]
    validation_input = h5f['validation_input'][:]
    validation_target = h5f['validation_target'][:]
    h5f.close()
    
    return training_input, training_target, validation_input, validation_target
def save_data(training_input, training_target, validation_input, validation_target, data_name, data_folder='./data'):
    h5f = h5py.File('{}/{}.h5'.format(data_folder, data_name), 'w')
    h5f.create_dataset('training_input', data=training_input)
    h5f.create_dataset('training_target', data=training_target)
    h5f.create_dataset('validation_input', data=validation_input)
    h5f.create_dataset('validation_target', data=validation_target)
    h5f.close()

def load_all_data(replayFolder, dataName, threadCount=8):   
    print("Loading data...")
    start = time.time()
    training_input = pandas.read_csv('./data/{}_training_input.csv'.format(dataName), sep=',')
    training_target = pandas.read_csv('./data/{}_training_target.csv'.format(dataName), sep=',')
    validation_input = pandas.read_csv('./data/{}_validation_input.csv'.format(dataName), sep=',')
    validation_target = pandas.read_csv('./data/{}_validation_target.csv'.format(dataName), sep=',')
    
    print("Finished loading data in {} seconds...".format(time.time() - start))
    return training_input, training_target, validation_input, validation_target
def load_all_replays(replayFolder, gamesToLoad, threadCount=8):   
    manager = multiprocessing.Manager()
    data_input = manager.Queue()
    data_target = manager.Queue()
    THREADCOUNT = threadCount
    replaySubsets = [[] for thread in range(THREADCOUNT)]
    replayNames = fetch_replay_names(replayFolder, gamesToLoad)
    for index, replay_name in enumerate(replayNames):
        replaySubsets[index % THREADCOUNT].append((index, replay_name))
        
    print("Generating training data...")
    start_time = time.time()
    threads = []
    lock = multiprocessing.Lock()
    max_width= multiprocessing.Value('i', 0)
    #max_width = multiprocessing.Manager().dict()
    for threadId in range(THREADCOUNT):
        threadArgs = (threadId, replayFolder, replaySubsets[threadId], data_input, data_target, lock,max_width,)
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
    
    validation_size = int(len(training_input)*.1)
    #validation_input = training_input[0:validation_size]
    #validation_target = training_target[0:validation_size]
    #training_input = training_input[validation_size:]
    #training_target = training_target[validation_size:]
    print("Finished loading {} games in {} seconds!".format(len(training_input), load_duration))
    
    training_input, validation_input, training_target, validation_target = train_test_split(training_input, training_target, test_size=0.1, random_state=1337)
    
    training_input = np.concatenate(training_input,axis=0)
    training_target = np.concatenate(training_target,axis=0)
    validation_input = np.concatenate(validation_input,axis=0)
    validation_target = np.concatenate(validation_target,axis=0)
    
    print("Finishing loading data, now beginning to save to disk...")
    print("Finished loading and saving data!")
    
    #move_counts = [0 for i in range(4)]
    """fuck = 0
    move_choices = training_target[:, 484:489]
    for i in range(len(move_choices)):
        move = move_choices[i].argmax()
        move_counts[move] += 1
        if (np.array_equal(move_choices[i], np.zeros(5))):
            fuck += 1
    """
    #print("---TABULATED MOVE COUNTS: ", move_counts, move_counts / np.sum(move_counts), fuck)
    return training_input, training_target, validation_input, validation_target


if __name__ == "__main__":
    if len(sys.argv) >= 5:
        USER_ID = int(sys.argv[1])
        REPLAY_FOLDER = sys.argv[2]
        REPLAY_LIMIT = int(sys.argv[3])
        THREAD_COUNT = int(sys.argv[4])
    X, y = load_all_replays("./replays", 5, 1)
    print(X.shape)
    print(y.shape)