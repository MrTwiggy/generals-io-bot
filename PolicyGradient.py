# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:56:25 2017

@author: MrTwiggy
"""

if __name__ == "__main__":
    import os, sys
    GPUS = sys.argv[1] if len(sys.argv) >= 2 else "0"
    os.environ["CUDA_VISIBLE_DEVICES"]=GPUS

from GameSimulator import simulate_game
from train_imitation import load_model_train, generate_batch
from LoadReplayData import fetch_replay_names
from game import MAX_MAP_WIDTH, ORIGINAL_MAP_WIDTH

import numpy as np
import sklearn
import keras
from keras import backend as K
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import clip_ops

import argparse
import sys
import glob
from random import randint
import time
import multiprocessing
from multiprocessing import Pool, Lock

def _to_tensor(x, dtype):
  x = ops.convert_to_tensor(x)
  if x.dtype != dtype:
    x = math_ops.cast(x, dtype)
  return x
# Loss for tile position:
# Given the predicted (23, 23) and actual (23, 23)
# Take the probability assigned to the actual truth tile chosen
# y_true: Concatenation of (23, 23) target for tile position, and a scalar discounted monte carlo reward
# y_pred: A (23, 23) prediction of tile positions
def tile_position_loss(y_true, y_pred):
    # y_pred: (n_samples, 23*23)
    # y_true: (n_samples, 23*23 + 1)
    batch_size = K.shape(y_true)[0]
    
    num_tiles = ORIGINAL_MAP_WIDTH**2
    y_true_tile_pos = K.reshape(y_true[:, :num_tiles], (batch_size, num_tiles)) # (batch_size, width^2)
    y_pred_tile_pos = K.reshape(y_pred, (batch_size, num_tiles))               # (batch_size, width^2)
    y_reward = y_true[:, num_tiles] #(batch_size)
    
    # The probability assigned to the true chosen tile by prediction
    y_pred_prob = K.sum(y_true_tile_pos * y_pred_tile_pos, axis=-1) # (batch_size)
    
    # Clip the predicted probability to remove exploding or vanishing gradient
    epsilon = _to_tensor(K.epsilon(), y_pred_prob.dtype.base_dtype)
    y_pred_prob = clip_ops.clip_by_value(y_pred_prob, epsilon, 1 - epsilon)
    
    y_pred_prob = K.log(y_pred_prob) * -1 # If we assign high probability, close to zero. Assign low probability, close to +inf
    
    # Without the below code, the loss should basically be behavioural cloning on itself
    y_pred_prob = y_pred_prob * y_reward
    y_pred_prob = K.mean(y_pred_prob)    
    
    return y_pred_prob

# Loss for tile position:
# Given the predicted (23, 23, 4) and actual (23, 23, 4)
# Take the probability assigned to the actual truth direction chosen
# y_true: Concatenation of (23, 23, 4) target for move direction, and a scalar discounted monte carlo reward
# y_pred: A (23, 23, 4) prediction of move_direction
def move_direction_loss(y_true, y_pred):
    # y_pred: (n_samples, 23*23*4)
    # y_true: (n_samples, 23*23*4 + 1)
    batch_size = K.shape(y_true)[0]
    
    num_tiles = 4 * ORIGINAL_MAP_WIDTH**2
    y_true_tile_pos = K.reshape(y_true[:, :num_tiles], (batch_size, num_tiles)) # (batch_size, width^2)
    y_pred_tile_pos = K.reshape(y_pred, (batch_size, num_tiles))               # (batch_size, width^2)
    y_reward = y_true[:, num_tiles] #(batch_size)
    
    # The probability assigned to the true chosen tile by prediction
    y_pred_prob = K.sum(y_true_tile_pos * y_pred_tile_pos, axis=-1) # (batch_size)
    
    # Clip the predicted probability to remove exploding or vanishing gradient
    epsilon = _to_tensor(K.epsilon(), y_pred_prob.dtype.base_dtype)
    y_pred_prob = clip_ops.clip_by_value(y_pred_prob, epsilon, 1 - epsilon)
    
    y_pred_prob = K.log(y_pred_prob) * -1 # If we assign high probability, close to zero. Assign low probability, close to +inf
    
    # Without the below code, the loss should basically be behavioural cloning on itself
    y_pred_prob = y_pred_prob * y_reward
    
    
    y_pred_prob = K.mean(y_pred_prob)    
    
    return y_pred_prob

class ModelPool(object):
    def __init__(self, model_name, max_opponents, data_folder):
        self.model_name = model_name
        self.max_opponents = max_opponents
        self.data_folder = data_folder
        self.version = 1
        self.load_models()
        self.updates = 0
    
    def load_models(self):
        # Load the main current model into self.current_model
        # Load opponent previous iteration models into self.opponent_models
        self.opponent_models = [self._load_policy_model(self.model_name)] # TODO: Look at all different versions under this name and pick the X most recent
        self.current_model = self._load_policy_model(self.model_name) #TODO: Search through all version with this name and pick most recent.
    
    def checkpoint_current_model(self, increment_version=True):
        model_name = "{}-v{}".format(self.model_name, self.version)
        model_path = "{}/{}.h5".format(self.data_folder, model_name)
        self.current_model.save(model_path)
        self._add_opponent(model_name)
        if increment_version:
            self.version += 1
    
    def train_current_model(self, batch_X, batch_y, discount):
        self.updates += 1
        
        #for rotation in range(4):
        #    for flip in range(2):
        #        copy_batch_X, copy_batch_y = np.copy(batch_X), np.copy(batch_y)
        #        copy_batch_X, copy_batch_y = generate_batch(copy_batch_X, copy_batch_y, True, self.updates, discount, 1, rotation, flip)      
        #        results = self.current_model.train_on_batch(copy_batch_X, copy_batch_y)
        copy_batch_X, copy_batch_y = np.copy(batch_X), np.copy(batch_y)
        copy_batch_X, copy_batch_y = generate_batch(copy_batch_X, copy_batch_y, True, self.updates, discount, 1)      
        results = self.current_model.train_on_batch(copy_batch_X, copy_batch_y)
        #print("Train duration: ", time.time() - start_time, " and for augmentation: ", augment_duration)
        return results
    
    def fetch_random_opponent(self):
        num_opponents = len(self.opponent_models)
        if num_opponents == 0:
            return self.current_model
        return self.opponent_models[randint(0, num_opponents - 1)]
    
    def _load_policy_model(self, model_name, custom=False):
        custom_objs = {'tile_position_loss':tile_position_loss, 'move_direction_loss':move_direction_loss} if custom else None
        model = load_model_train(self.data_folder, model_name, custom_objs)
        
        model_loss = {'tile_position':tile_position_loss, 'move_direction': move_direction_loss}
        model_weighting = {'tile_position': 1.0, 'move_direction' : 1.0}
        model.compile(keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss=model_loss, loss_weights=model_weighting)
        return model
        
    def _add_opponent(self, model_name):
        model = self._load_policy_model(model_name, True)
        self.opponent_models.append(model)
        
        if len(self.opponent_models) > self.max_opponents:
            self.opponent_models.pop(0)
    
if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Process PolicyGradient training arguments')
    #parser.add_argument('model_name', action="store", dest="model_name")
    #parser.add_argument('--gamesperbatch', action="store", dest="gamesperbatch", type=int, default=5)
    #
    #args = parser.parse_args()
    #print(args)
    REPLAY_FOLDER = "./replays/Pros"
    MODEL_NAME = "CurrentBest"
    GAMES_PER_BATCH = 1
    ITERATIONS_PER_VERSION = 200
    MAX_OPPONENTS = 10
    DATA_FOLDER = "./data"
    GPUS = sys.argv[1] if len(sys.argv) >= 2 else "0"
    MODEL_NAME = sys.argv[2] if len(sys.argv) >= 3 else "CurrentBest"
    SEED = int(sys.argv[3]) if len(sys.argv) >= 4 else 1337
    np.random.seed(SEED)
    DISCOUNT=0.995
    TRAIL_LENGTH = 75
    BATCH_SIZE = 164
    
    MODEL_POOL = ModelPool(MODEL_NAME, MAX_OPPONENTS, DATA_FOLDER)
    # TODO: Initialize all of the above using parsed arguments from argparser
    
    trailing_wins = []
    trailing_turns = []
    win_rate = 0
    turn_rate = 0
    # Loop indefinitely 
    while True:
        for iteration in range(ITERATIONS_PER_VERSION):
            print("-------Beginning Iteration {}---------".format(iteration))
            batch_X, batch_y = [], []
            game_count = 0
            total_games = 0
            main_wins = 0
            start_time = time.time()
            while game_count < GAMES_PER_BATCH:
                iteration_duration = time.time() - start_time
                print("Finished games {} out of {} in {} seconds so far with {} wins and winrate {} and turn-rate {}...".format(game_count, total_games, iteration_duration, main_wins, win_rate, turn_rate))
                total_games += 1
                replay_name = fetch_replay_names(REPLAY_FOLDER, 1, 2)[0]
                main_id = randint(0, 1)
                opponent_id = 1 - main_id
                models = [None, None]
                models[main_id] = MODEL_POOL.current_model
                models[opponent_id] = MODEL_POOL.fetch_random_opponent()
                winner, forced_finish, match_inputs, match_targets, turns = simulate_game(game_count, models, replay_name, 1.0, 800, REPLAY_FOLDER, False)
                
                if forced_finish: # We don't want to include games with a forced finish
                    a = 1
                    continue
                
                if winner == main_id:
                    main_wins += 1
                
                game_count += 1
                
                trailing_wins.append(int(winner == main_id))
                trailing_turns.append(turns)
                if len(trailing_wins) > TRAIL_LENGTH:
                    trailing_wins.pop(0)
                    trailing_turns.pop(0)
                win_rate = float(sum(trailing_wins)) / len(trailing_wins)
                turn_rate = float(sum(trailing_turns)) / len(trailing_turns)
                
                main_inputs, main_targets = match_inputs[main_id].astype('float32'), match_targets[main_id].astype('float32')
                #print(main_inputs.dtype, main_targets.dtype, "------------------")
                batch_X.append(main_inputs)
                batch_y.append(main_targets)
            
            batch_X = np.concatenate(batch_X, axis=0).astype('float32')
            batch_y = np.concatenate(batch_y, axis=0).astype('float32')
            print("Finished data collection for iteration {}, beginning model mini-batch update of shape {}.".format(iteration, batch_X.shape))
            
            #batch_X, batch_y = sklearn.utils.shuffle(batch_X, batch_y)
            n_samples = len(batch_X)
            n_batches = 1#n_samples // BATCH_SIZE + (1 if n_samples%BATCH_SIZE != 0 else 0)
            results = np.array([0.0, 0.0, 0.0])
            for i in range(n_batches):
                start = max(0, n_samples - BATCH_SIZE)#i*BATCH_SIZE
                end = min(n_samples, start+BATCH_SIZE)
                result = np.array(MODEL_POOL.train_current_model(np.array(batch_X[start:end]), np.array(batch_y[start:end]), DISCOUNT))
                #print(result)
                results += result
            
            print("Results: ", result / n_batches)
            #MODEL_POOL.checkpoint_current_model(False)
            #tf.reset_default_graph()
            print("Finished entire iteration #{} in {} seconds!".format(iteration, time.time() - start_time))
                
        # Checkpoint the current model version and add it to the opponent pool
        MODEL_POOL.checkpoint_current_model()
        