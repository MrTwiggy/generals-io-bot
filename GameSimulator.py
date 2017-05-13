# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:28:51 2017

@author: MrTwiggy
"""


import sys
if __name__ == "__main__":
    if len(sys.argv) >= 5:
        SIMULATION_NAME = sys.argv[1]
        REPLAY_FOLDER = sys.argv[2]
        MODEL_NAME1 = sys.argv[3]
        MODEL_NAME2 = sys.argv[4]
        MODEL_NAMES = [MODEL_NAME1, MODEL_NAME2]
        GAME_COUNT = int(sys.argv[5])
        MAX_TURN_LIMIT = int(sys.argv[6]) if len(sys.argv) >= 7 else 250
        GPUS = sys.argv[7] if len(sys.argv) >= 8 else "0,1"
        
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]=GPUS

import numpy as np
import time
import h5py
import game as generals_game
import os
from random import randint

from sklearn.model_selection import train_test_split
from bot_TNT import update_state, calculate_action
from LoadReplayData import fetch_replay_names, load_replay, generate_blank_state, generate_target, coordinates_to_direction, generate_target_tensors
from train_imitation import load_model_train

def simulate_game(game_id, models, replay_name, discount = 1.0, max_turns=500, replay_folder="./replays", verbose=False):
    #print(models[0].summary())
    #print(models[1].summary())
    #os.environ["CUDA_VISIBLE_DEVICES"]="1"
    forced_finish = False
    replay = load_replay(replay_folder, replay_name)
    game = generals_game.Game.from_replay(replay)
    game_states = [generate_blank_state(), generate_blank_state()]
    stats = [None, None]
    height = game.gmap.height
    width = game.gmap.width
    
    game_inputs = [[], []]
    game_targets = [[], []]
    last_moves = [None, None]
    internal_states = [None, None]
    
    while not game.is_over():
        if verbose:
            print("---------------Game #{}, turn {}-----------------------".format(game_id, game.turn))             
        
        # Update each bot's GameState
        for i in range(2):
            enemy = 0 if i == 1 else 1
            tiles, armies, cities, generals = game.generate_state(i)
            internal_states[i] = (tiles, armies, cities, generals)
            tiles = tiles.reshape(height, width)
            armies = armies.reshape(height, width)
            
            oracle_tiles, oracle_armies, oracle_cities, oracle_generals = game.generate_state(i, oracle=True)
            oracle_tiles = oracle_tiles.reshape(height, width)
            oracle_armies = oracle_armies.reshape(height, width)
            enemy_stats = (np.sum(oracle_armies[oracle_tiles == enemy]), np.sum(oracle_tiles == enemy))
            player_stats = (np.sum(oracle_armies[oracle_tiles == i]), np.sum(oracle_tiles == i))
            #stats[i] = player_stats
            game_states[i] = update_state(game_states[i], game.turn, tiles, armies, cities, generals, i, enemy, player_stats, enemy_stats, last_moves[i])
        
        # Let each bot perform an action
        for i in range(2):
            enemy = 0 if i == 1 else 1
            tiles, armies, cities, generals = internal_states[i]
            tiles = tiles.reshape(height, width)
            armies = armies.reshape(height, width)
            current_state = np.copy(game_states[i])
            state_copy = np.copy(current_state)
            action = calculate_action(models[i], current_state, game.turn, tiles, armies, i, enemy, i, verbose=verbose)
            
            if action is not None:
                y, x, y_dest, x_dest, y_padding, x_padding = action
                if verbose:
                    print("Player ", i, " moved from ", y, x, " to ", y_dest, x_dest)
                success = game.handle_attack(i, game.gmap.index_for(y, x), game.gmap.index_for(y_dest, x_dest), False)
                
                if success:
                    direction = coordinates_to_direction(y, x, y_dest, x_dest)
                    last_moves[i] = generate_target_tensors(x, y, direction)
                    
                    if np.random.binomial(1, p=0.75):
                        game_inputs[i].append(state_copy)
                        game_targets[i].append(generate_target(game, y, x, y_dest, x_dest))
                else:
                    if verbose:
                        print("Error: Submitted move was unsuccessful...")
                    
            else:
                if verbose:
                    print("Unable to submit move for player ", i)
        
        # Update the game state at the end of the turn
        game.update()
        
        if game.turn >= max_turns:
            game.end_game()
            forced_finish = True
            break
    
    
    winner = game.winner()
    if verbose:
        print("Finished game #{} with winner {}".format(game_id, winner))
    
    for player in range(2):
        for i in range(len(game_inputs[player])):
            target = 1 if player == winner else -1
            turns_left = game.turn - game_targets[player][i][6]
            #print(target, " vs ", float(target) * (discount ** turns_left))
            #print(game_targets[player][i])
            game_targets[player][i] = game_targets[player][i]
            game_targets[player][i][5] = float(target) * (discount ** turns_left)
            game_targets[player][i][6] = turns_left
        
        game_inputs[player] = np.array(game_inputs[player])
        game_targets[player] = np.array(game_targets[player])
        #print(game_inputs[player].shape, game_targets[player].shape)
    #for i in range(2):
    #    print(game_inputs[i].shape)
    game_inputs = np.array(game_inputs) # (2, #samples, 23, 23, 38)
    game_targets = np.array(game_targets) # (2, #samples, 7)
    
    return winner, forced_finish, game_inputs, game_targets, game.turn

if __name__ == "__main__":
    models = [load_model_train("./data", MODEL_NAMES[0]), load_model_train("./data", MODEL_NAMES[1])]
    games = 0
    results = [0, 0]
    forced_finishes = 0
    replay_names = fetch_replay_names(REPLAY_FOLDER, GAME_COUNT, 2)
    print("Loaded {} replay map starting points!".format(len(replay_names)))
    #time.sleep(5)
    
    start_time = time.time()
    training_input = []
    training_target = []
    for replay_name in replay_names:
        games += 1
        print("Replay ", replay_name)
        
        player1_id = randint(0, 1)
        player2_id = 1 - player1_id
        player_ids = [player1_id, player2_id]
        
        game_models = [models[player_id] for player_id in player_ids]
        winner, forced_finish, match_inputs, match_targets, turns = simulate_game(games, game_models, replay_name, max_turns=MAX_TURN_LIMIT, replay_folder=REPLAY_FOLDER, verbose=False)
        #training_input.append(match_inputs)
        #training_target.append(match_targets)
        results[player_ids[winner]] += 1
        forced_finishes += forced_finish
        print("Finished new game! Current win results: {} vs {} in match {} vs {} for a winrate {} and {} forced finishes...".format(results[0], results[1], MODEL_NAMES[0], MODEL_NAMES[1], results[0] / float(results[0] + results[1]), forced_finishes))
    print("Finished simulating {} games in {} seconds with {} forced game ends!".format(games, time.time() - start_time, forced_finishes))
    
    """training_input= np.array(training_input)
    training_target= np.array(training_target)
    print(training_input.shape, training_target.shape)
    training_input, validation_input, training_target, validation_target = train_test_split(training_input, training_target, test_size=0.1, random_state=1337)
    
    training_input = np.concatenate(training_input,axis=0)
    training_target = np.concatenate(training_target,axis=0)
    validation_input = np.concatenate(validation_input,axis=0)
    validation_target = np.concatenate(validation_target,axis=0)
    
    print("Shapes: ", training_input.shape, training_target.shape, validation_input.shape, validation_target.shape)
    print("Saving...")
    load_time = time.time()    
    save_data(training_input, training_target, validation_input, validation_target, SIMULATION_NAME)
    print("Finished saving with h5py in {} seconds!".format(time.time() - load_time))
    
    load_time = time.time()
    training_input, training_target, validation_input, validation_target = load_data(SIMULATION_NAME)
    print("Finished loading with h5py in {} seconds!".format(time.time() - load_time))
    print(training_input.shape, training_target.shape, validation_input.shape, validation_target.shape)"""
    