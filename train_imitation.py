# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 17:30:42 2017

@author: MrTwiggy
"""

import datetime
import os
import sys

import numpy as np
import math
import json
import time

import multiprocessing

import traceback
import keras
from keras import backend as K
from keras.models import Sequential,load_model, Model
from keras.layers import Dense, Convolution2D, Activation, Flatten, Input, Lambda, Reshape, Dropout, merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.backend.common import _EPSILON
from keras.backend.tensorflow_backend import _to_tensor
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
from random import randint
from game import MAX_MAP_WIDTH, ORIGINAL_MAP_WIDTH

from LoadReplayData import load_all_replays, generate_target_tensors

from sklearn.utils import shuffle

import tensorflow as tf

#from fetch_player_data import download_player_replays

#download_player_replays(3146, "./replays/forest", 6000)

# python ./train_imitation.py
if __name__ == "__main__":
    arg_count = len(sys.argv) - 1
    
    REPLAY_FOLDER = sys.argv[1] if arg_count >= 1 else "./replays"
    THREAD_COUNT = int(sys.argv[2]) if arg_count >= 2 else 8
    MODEL_NAME = sys.argv[3] if arg_count >= 3 else "default-model"
    MAX_GAMES_TO_LOAD = int(sys.argv[4]) if arg_count >= 4 else 100
training_input = []
training_target = []

DATA_FOLDER = "./data"
BATCH_SIZE = 64


def multi_label_accuracy(y_true, y_pred):
    batch_size = K.shape(y_true)[0]
    y_true = K.reshape(y_true, (batch_size, ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, 4)) # (batch_size, 2500, 5)
    y_pred = K.reshape(y_pred, (batch_size, ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, 4)) # (batch_size, 2500, 5)
    move_mask = K.cast(K.any(y_true, axis=-1), dtype='float32') #(batch_size, 23, 23)
    y_args = K.argmax(y_true, axis=-1) + 1 # y_args = (batch_size, 23, 23)
    y_args = K.cast(y_args, dtype='float32')
    y_args = tf.multiply(y_args, move_mask)
    
    y_args_pred = K.argmax(y_pred, axis=-1) + 1 # y_args_pred = (batch_size, 23, 23)
    y_args_pred = K.cast(y_args_pred, dtype='float32')
    y_args_pred = tf.multiply(y_args_pred, move_mask)
    accuracies = []
    for MOVE_INDEX in range(1, 5):
        move_counts = K.cast(tf_count(K.reshape(y_args, (batch_size, ORIGINAL_MAP_WIDTH**2)), MOVE_INDEX), dtype='float32') # (batch_size)
        move_counts = K.sum(move_counts) #(1)
        #move_counts = tf.maximum(move_counts, 1)
        true_stills = K.cast(K.equal(y_args, MOVE_INDEX), dtype='float32') #(batch_size, 23, 23)
        pred_stills = K.cast(K.equal(y_args_pred, MOVE_INDEX), dtype='float32') #(batch_size, 23, 23)
        corrects = K.sum(K.sum(tf.multiply(true_stills, pred_stills), axis=-1), axis=-1) # (batch_size)
        result = tf.div(K.sum(corrects), tf.maximum(move_counts, 1))
        #result = corrects
        #result = K.mean(result)
        #result = K.mean(move_counts)
        #result = tf.Print(result, [y_args, result, corrects, move_counts],message="MESSAGE: ", summarize=32)
        accuracies.append(result)
    return {'North': accuracies[0], 
            'East': accuracies[1], 
            'South': accuracies[2], 
            'West': accuracies[3]}


def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints, axis=-1)
    return count

def multi_label_crossentropy(y_true, y_pred):
    # Target input: (batch_size, 2420)
    batch_size = K.shape(y_true)[0]

    true_shaped = K.reshape(y_true, (batch_size, ORIGINAL_MAP_WIDTH,ORIGINAL_MAP_WIDTH, 4)) #(batch_size, 22, 22, 5)
    true_flat = K.reshape(y_true, (batch_size*ORIGINAL_MAP_WIDTH*ORIGINAL_MAP_WIDTH, 4))
    pred_flat = K.reshape(y_pred, (batch_size*ORIGINAL_MAP_WIDTH*ORIGINAL_MAP_WIDTH, 4))
    
    move_mask = K.any(K.reshape(y_true, (batch_size, ORIGINAL_MAP_WIDTH*ORIGINAL_MAP_WIDTH, 4)), axis=-1)#(batch_size, 50,50)
    #move_counts = K.sum(move_mask, axis=-1)#(batch_size)
    move_counts = tf_count(move_mask, 1)
    
    pred_flat /= tf.reduce_sum(pred_flat, reduction_indices=len(pred_flat.get_shape()) - 1, keep_dims=True)
    # manual computation of crossentropy
    epsilon = _to_tensor(_EPSILON, pred_flat.dtype.base_dtype)
    pred_flat = tf.clip_by_value(pred_flat, epsilon, 1. - epsilon)
    test1 = -(true_flat * tf.log(pred_flat))
    crossentropy = tf.reduce_sum(test1,
                           reduction_indices=len(pred_flat.get_shape()) - 1)
                               
    crossentropy = K.reshape(crossentropy, (batch_size, ORIGINAL_MAP_WIDTH,ORIGINAL_MAP_WIDTH)) # (batch_size, 2500)
    crossentropy = K.sum(K.sum(crossentropy, axis=-1), axis=-1) #(batch_size)
    before = crossentropy
    
    crossentropy = tf.div(crossentropy, K.cast(move_counts, dtype='float32'))
    #crossentropy = tf.Print(crossentropy, [y_true, y_pred, crossentropy], message="Testing: ", summarize=5)
    return crossentropy

#batch_X shape: (batch_size, map_height, map_width, 11)
#batch_y shape is tuple ((y,x), direction)

def generate_paddings(height, width):
    diff_x = float(ORIGINAL_MAP_WIDTH - width) / 2.0
    diff_y = float(ORIGINAL_MAP_WIDTH - height) / 2.0
    
    return (math.ceil(diff_y), math.floor(diff_y)), (math.ceil(diff_x), math.floor(diff_x)),
def augment_batch(batch_X, batch_y, batch_size):
    for i in range(batch_size):
        rotation = randint(0, 3)
        flip_vert = randint(0,1)
        #y, x, direction, height, width = batch_y[i]
        #y_padding, x_padding = generate_paddings(height, width)
        
        example_tile = np.copy(batch_y[i, :ORIGINAL_MAP_WIDTH**2]).reshape(ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH)
        example_direction = np.copy(batch_y[i, ORIGINAL_MAP_WIDTH**2:]).reshape(ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, 4)
        
        start_y, start_x = np.unravel_index(np.argmax(example_tile.flatten()), example_tile.shape)
        start_tile = np.copy(batch_X[i][start_y, start_x])
        
        direction = np.argmax(example_direction[example_tile == 1])
        start_direction = direction
        
        # Rotate the map input by the random amount chosen
        #print("Batch X Shape: ", batch_X.shape, batch_X[i].shape)
        batch_X[i] = np.rot90(np.copy(batch_X[i]), k=rotation, axes=(1,0))
        example_tile = np.rot90(np.copy(example_tile), k=rotation, axes=(1,0))
        
        direction = (direction + rotation) % 4
        
        # Flip the map vertically if decided
        if flip_vert:
            batch_X[i] = np.flipud(np.copy(batch_X[i]))
            example_tile = np.flipud(np.copy(example_tile))
            #example_direction = np.flipud(example_direction)
            
            if direction == 0:
                direction = 2
            elif direction == 2:
                direction = 0
        
        final_direction = np.zeros((ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, 4))
        final_direction[example_tile == 1, direction] = 1    
        
        batch_y[i, :ORIGINAL_MAP_WIDTH**2] = example_tile.flatten()
        batch_y[i, ORIGINAL_MAP_WIDTH**2:] = final_direction.flatten()
        
        
        end_y, end_x = np.unravel_index(np.argmax(example_tile.flatten()), example_tile.shape)
        end_direction = np.argmax(final_direction[end_y, end_x])
        end_tile = batch_X[i][end_y, end_x]
        #print("Tiles ", end_tile, " vs ", start_tile)
        #print("Start: ", start_y, start_x, " VS End: ", end_y, end_x, " with shapes ", example_tile.shape, final_direction.shape, start_direction, end_direction)
        
                        
        
    
    batch_X = np.concatenate(np.array([batch_X]), axis=0)
    batch_y = np.concatenate(np.array([batch_y]), axis=0)
    
    for i in range(len(batch_y)):
        batch_y[i] = batch_y[i].astype('float32')
    batch_y = np.array([batch_y[i] for i in range(len(batch_y))], dtype='float32')
    batch_y = batch_y.reshape(batch_size, -1)
    
    for i in range(len(batch_y)):
        batch_X[i] = batch_X[i].astype('float32')
    batch_X = np.array([batch_X[i] for i in range(len(batch_X))], dtype='float32')
    batch_X = batch_X.reshape(batch_size, ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, 11)
    
    return batch_X, batch_y

# training_input_orig is of shape (num_samples, map_height, map_width, 11)
# training_target_orig is of shape (num_samples, 1),
# each row in target is a tuple of ((y,x), direction) where
# x,y are the positions of the moving tile and direction is an int in [0,3] denoting the direction moved
def frame_generator(training_input_orig, training_target_orig, batch_size, augment=False):
    while True:
        #print("----------SHUFFLING FRAMES!")
        training_input, training_target = shuffle(training_input_orig, training_target_orig)
        batches = math.ceil(float(len(training_input)) / batch_size)
        
        for batch_index in range(batches):
            start = batch_index*batch_size
            end = min(len(training_input), start + batch_size)
            batch_size = end - start
            
            batch_X, batch_y = np.copy(training_input[start:end]), np.copy(training_target[start:end])
            
            #batch_y = batch_y[:, :529]
            tile_target = []
            move_direction_target = []
            
            for i in range(batch_size):
                y, x, direction, height, width = batch_y[i]
                #print(i, y, x, direction, height, width)
                tile_pos, move_direction = generate_target_tensors(x, y, direction)
                tile_target.append(tile_pos.flatten())
                move_direction_target.append(move_direction.flatten())
            
            tile_target = np.array(tile_target).astype('float32')
            move_direction_target = np.array(move_direction_target).astype('float32')
            
            tile_len = tile_target.shape[1]
            move_len = move_direction_target.shape[1]
            
            batch_y = np.concatenate((tile_target, move_direction_target), axis=1)
            #print("------- batch shape ", batch_y.shape)
            
            if augment:
                batch_X, batch_y = augment_batch(batch_X, batch_y, batch_size)
                #batch_X, batch_y = augment_batch(batch_X, batch_y, batch_size)
            
            
                
            
            yield batch_X, [batch_y[:, :tile_len], batch_y[:, tile_len:]]

def build_model():
    #---- Shared convolution network 
    main_input = Input(shape=(ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, 11,), dtype='float32', name='main_input')
    """cnn = Convolution2D(128, 3, 3, border_mode="same",activation = 'relu')(main_input)
    cnn = BatchNormalization()(cnn)
    cnn = Convolution2D(128, 3, 3, border_mode="same", activation = 'relu')(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Convolution2D(96, 3, 3, border_mode="same", activation = 'relu')(cnn)
    cnn = BatchNormalization()(cnn)
    #cnn = Convolution2D(128, 3, 3, border_mode="valid", activation = 'relu')(cnn)
    #cnn = BatchNormalization()(cnn)
    """
    cnn = Convolution2D(96, 7, 7, border_mode="same",activation = 'relu')(main_input)
    cnn = BatchNormalization()(cnn)
    cnn = Convolution2D(64, 5, 5, border_mode="same", activation = 'relu')(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Convolution2D(32, 3, 3, border_mode="same", activation = 'relu')(cnn)
    cnn = BatchNormalization()(cnn)
    
    #---- Tile position network
    #tile_position = Convolution2D(64, 3, 3, border_mode="same", activation = 'relu')(cnn)
    #tile_position = BatchNormalization()(tile_position)
    tile_position = Convolution2D(1, 5, 5, border_mode="same", activation='relu')(cnn)
    tile_position = Flatten()(tile_position)
    tile_position = Activation('softmax', name='tile_position')(tile_position)  
    
    #------ Move direction network
    #move_direction = Convolution2D(64, 3, 3, border_mode="same", activation = 'relu')(cnn)
    #move_direction = BatchNormalization()(move_direction)
    move_direction = Convolution2D(4, 5, 5, border_mode="same", activation='relu')(cnn)
    move_direction = Reshape((ORIGINAL_MAP_WIDTH*ORIGINAL_MAP_WIDTH, 4))(move_direction)
    move_direction = Lambda(lambda x: tf.nn.softmax(x))(move_direction)
    move_direction = Flatten(name='move_direction')(move_direction)    
    
    #----- Is half-move network
    #is50 = Dense(1, activation='sigmoid', name='is50')(merge([flattened_cnn, tile_position, move_direction], mode='concat'))
    
    #---- Final built model
    model = Model(input=main_input, output=[tile_position, move_direction])#, is50])
    model.compile('rmsprop', loss={'tile_position': 'categorical_crossentropy', 'move_direction': multi_label_crossentropy},
                  metrics={'move_direction': multi_label_accuracy, 'tile_position':'accuracy'},loss_weights={'tile_position': 1, 'move_direction' : 2.0})
    
    return model

def load_model_train(dataFolder, modelName):
    """tile_model = build_tile_model()
    tile_model.load_weights('{}_tile/{}.h5'.format(dataFolder, modelName))
    direction_model = build_direction_model()
    direction_model.load_weights('{}_direction/{}.h5'.format(dataFolder, modelName))
    return tile_model, direction_model"""
    model = build_model()
    model.load_weights('{}/{}.h5'.format(dataFolder, modelName))
    return model

def shape_and_pad(training_in):
    a = training_in.reshape(-1, ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, 1)
    print(a.shape)
    a = np.pad(a, ((0,0), (4, 4), (4, 4), (0,0)), mode='constant', constant_values=0)
    return a

# --------------------- Main Training Logic -----------------------
if __name__ == "__main__":
    print("----STARTING TRAININGv2------")
    np.random.seed(1337) # for reproducibility
    
    training_input, training_target, validation_input, validation_target = load_all_replays(REPLAY_FOLDER, MAX_GAMES_TO_LOAD, THREAD_COUNT)
    model = build_model()
    
    for epoch in range(1,30):
        print("Initializing meta epoch: ", epoch)
        print(model.summary())
        checkpoint_name = '{}/{}v{}.h5'.format(DATA_FOLDER, MODEL_NAME, epoch)
        print("Checkpoint name: ", checkpoint_name)
        now = datetime.datetime.now()
        tensorboard = TensorBoard(log_dir='./logs/'+now.strftime('%Y.%m.%d %H.%M'))
        
        print("Training shapes: ", training_target.shape, training_input.shape)
        print("Validation shapes: ", validation_target.shape, validation_input.shape)
        tile_train_target = training_target[:, :ORIGINAL_MAP_WIDTH**2]
        tile_validation_target = validation_target[:, :ORIGINAL_MAP_WIDTH**2]
        
        model.fit_generator(
            frame_generator(training_input, training_target, BATCH_SIZE, augment=True),
            validation_data=frame_generator(validation_input, validation_target, BATCH_SIZE, augment=False),
            samples_per_epoch=len(training_input),
            nb_val_samples=len(validation_input),
            nb_epoch=10,
            verbose=1,
            callbacks=[EarlyStopping(patience=8), tensorboard,
                       ModelCheckpoint(checkpoint_name,verbose=1,save_best_only=True)])
        # --- Train the move direction model
        """shaped_tile_train = shape_and_pad(tile_train_target)
        shaped_tile_validation = shape_and_pad(tile_validation_target)
        print(shaped_tile_train.shape)
        direction_train_input = np.concatenate((training_input, shaped_tile_train), axis=3)
        direction_validation_input = np.concatenate((validation_input, shaped_tile_validation), axis=3)
        direction_train_target = training_target[:, 529:]
        direction_validation_target = validation_target[:, 529:]
        direction_model.fit_generator(
            frame_generator(direction_train_input, direction_train_target, BATCH_SIZE, augment=False),
            validation_data=frame_generator(direction_validation_input, direction_validation_target, BATCH_SIZE, augment=False),
            samples_per_epoch=len(training_input),
            nb_val_samples=len(validation_input),
            nb_epoch=10,
            verbose=1,
            callbacks=[EarlyStopping(patience=8), tensorboard,
                       ModelCheckpoint(checkpoint_name,verbose=1,save_best_only=True)])
        
        # --- Train the tile position model
        
        tile_model.fit_generator(
            frame_generator(training_input, tile_train_target, BATCH_SIZE),
            validation_data=frame_generator(validation_input, tile_validation_target, BATCH_SIZE, augment=False),
            samples_per_epoch=len(training_input),
            nb_val_samples=len(validation_input),
            nb_epoch=10,
            verbose=1,
            callbacks=[EarlyStopping(patience=8), tensorboard,
                       ModelCheckpoint(checkpoint_name,verbose=1,save_best_only=True)])"""
        
        
        