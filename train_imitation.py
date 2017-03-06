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
from game import MAX_MAP_WIDTH

from LoadReplayData import load_all_replays

from sklearn.utils import shuffle

import tensorflow as tf

#from fetch_player_data import download_player_replays

#download_player_replays(3146, "./replays/forest", 6000)

arg_count = len(sys.argv) - 1


REPLAY_FOLDER = sys.argv[1] if arg_count >= 1 else "./replays"
THREAD_COUNT = int(sys.argv[2]) if arg_count >= 2 else 8
MODEL_NAME = sys.argv[3] if arg_count >= 3 else "default-model"
DATA_NAME = sys.argv[4] if arg_count >= 4 else "default-data"
training_input = []
training_target = []

print("----STARTING TRAINING------")

DATA_FOLDER = "./data"
#MAX_MAP_WIDTH = 68
ORIGINAL_MAP_WIDTH = 50
MAP_DEPTH = 5
MAX_GAMES_TO_LOAD = 6000
BATCH_SIZE = 32
MOVE_WEIGHTING = 1
GAME_SAMPLE_RATE = 70 # The percentage of frames from each game to sample
MIN_PROB = 0.25
EARLY_GAME_PROBS = np.array([max(2.0/(index+1), MIN_PROB) for index in range(75)])



def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints, axis=-1)
    return count

"""
STILL = 0
NORTH = 1
EAST = 2
SOUTH = 3
WEST = 4
"""
def multi_label_accuracy(y_true, y_pred):
    batch_size = K.shape(y_true)[0]
    y_true = K.reshape(y_true, (batch_size, ORIGINAL_MAP_WIDTH*ORIGINAL_MAP_WIDTH, 5)) # (batch_size, 2500, 5)
    y_pred = K.reshape(y_pred, (batch_size, ORIGINAL_MAP_WIDTH*ORIGINAL_MAP_WIDTH, 5)) # (batch_size, 2500, 5)
    y_args = K.argmax(y_true, axis=-1)
    accuracies = []
    for MOVE_INDEX in range(5):
        #move_mask = K.any(y_true, axis=-1) #(batch_size, 2500)
        move_counts = K.sum(K.cast(tf_count(y_args, MOVE_INDEX), dtype='float32')) #(batch_size)
        move_counts = tf.maximum(move_counts, 1)
        true_stills = K.cast(K.equal(y_args, MOVE_INDEX), dtype='float32') #(batch_size, 2500)
        pred_stills = K.cast(K.equal(K.argmax(y_pred, axis=-1), MOVE_INDEX), dtype='float32') #(batch_size, 2500)
        corrects = K.sum(K.sum(tf.mul(true_stills, pred_stills), axis=-1))
        result = tf.div(corrects, move_counts)
        result = K.mean(result)
        #result = tf.Print(result, [y_args, result, corrects, move_counts],message="MESSAGE: ", summarize=32)
        accuracies.append(result)
    return {'Still': accuracies[0], 
            'North': accuracies[1], 
            'East': accuracies[2], 
            'South': accuracies[3], 
            'West': accuracies[4]}

def multi_label_crossentropy(y_true, y_pred):
    batch_size = K.shape(y_true)[0]

    true_shaped = K.reshape(y_true, (batch_size, ORIGINAL_MAP_WIDTH,ORIGINAL_MAP_WIDTH, 5)) #(batch_size, 2500, 5)
    true_flat = K.reshape(y_true, (batch_size*ORIGINAL_MAP_WIDTH*ORIGINAL_MAP_WIDTH, 5))
    pred_flat = K.reshape(y_pred, (batch_size*ORIGINAL_MAP_WIDTH*ORIGINAL_MAP_WIDTH, 5))
    
    move_mask = K.any(K.reshape(y_true, (batch_size, ORIGINAL_MAP_WIDTH*ORIGINAL_MAP_WIDTH, 5)), axis=-1)#(batch_size, 50,50)
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
    
    return crossentropy

def build_model():
    main_input = Input(shape=(MAX_MAP_WIDTH, MAX_MAP_WIDTH, 11,), dtype='float32', name='main_input')
    cnn = Convolution2D(256, 3, 3, border_mode="valid", input_shape=(MAX_MAP_WIDTH, MAX_MAP_WIDTH, 11,) ,activation = 'relu')(main_input)
    cnn = BatchNormalization()(cnn)
    cnn = Convolution2D(196, 3, 3, border_mode="valid", activation = 'relu')(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Convolution2D(128, 3, 3, border_mode="valid", activation = 'relu')(cnn)
    cnn = BatchNormalization()(cnn)
    #tile_position = BatchNormalization(cnn)
    tile_position = Convolution2D(1, 3, 3, border_mode="valid", activation='relu')(cnn)
    tile_position = Flatten()(tile_position)
    tile_position = Activation('softmax', name='tile_position')(tile_position)
    #tile_position = Dense(484, activation='softmax', name='tile_position')(cnn)
    #tile_position = Softmax()(tile_position)
    flattened_cnn = Flatten()(cnn)
    move_direction = Dense(5, activation='relu')(merge([flattened_cnn, tile_position], mode='concat'))
    move_direction = Activation('softmax', name='move_direction')(move_direction)
    #move_direction = Softmax()(move_direction)
    is50 = Dense(1, activation='sigmoid', name='is50')(merge([flattened_cnn, tile_position, move_direction], mode='concat'))
    
    model = Model(input=main_input, output=[tile_position, move_direction, is50])
    #model = Model(input=[main_input], output=[tile_position, move_direction, is50])
    return model

def load_model_train(dataFolder, modelName):
    model = build_model()
    model.compile('rmsprop', loss={'tile_position': 'categorical_crossentropy', 'move_direction': 'categorical_crossentropy','is50': 'binary_crossentropy'},
              loss_weights={'tile_position': 1, 'move_direction': 1,'is50': 1})
    model.load_weights('{}/{}.h5'.format(dataFolder, modelName))
    return model

if __name__ == "__main__":
    np.random.seed(133737) # for reproducibility
    training_input, training_target = load_all_replays("./replays", MAX_GAMES_TO_LOAD, 4)
        
    if True:
        model = build_model()
                            
        model.compile('rmsprop', loss={'tile_position': 'categorical_crossentropy', 'move_direction': 'categorical_crossentropy','is50': 'binary_crossentropy'},
              loss_weights={'tile_position': 1, 'move_direction': 1,'is50': 1})
        #model.load_weights('{}/{}.h5'.format(DATA_FOLDER, MODEL_NAME))
    else:
        a = 1
        print('{}/{}.h5'.format(DATA_FOLDER, MODEL_NAME))
        model = load_model('{}/{}.h5'.format(DATA_FOLDER, MODEL_NAME), {'multi_label_crossentropy' : multi_label_crossentropy})
    print('done')
    
    epoch = 1
    
    print("----START CHECK")
    
    print("----END CHECK")
    while epoch < 30:
        print("Training for epoch: ", epoch)
        print(model.summary())
        checkpoint_name = '{}/{}v{}.h5'.format(DATA_FOLDER, MODEL_NAME, epoch)
        print("Checkpoint name: ", checkpoint_name)
        #print(training_target[5][700:750])
        now = datetime.datetime.now()
        tensorboard = TensorBoard(log_dir='./logs/'+now.strftime('%Y.%m.%d %H.%M'))
        
        """model.fit_generator(
            frame_generator(training_input, training_target, BATCH_SIZE),
            samples_per_epoch=len(training_input)*1,
            nb_epoch=10,
            verbose=1,
            callbacks=[EarlyStopping(patience=10),
                             ModelCheckpoint(checkpoint_name,verbose=1,save_best_only=True),
                             tensorboard],
            validation_data=frame_generator(validation_input, validation_target, BATCH_SIZE, augment=False),
            nb_val_samples=len(validation_input)
        )"""
        #print(training_target[:, :484].shape, training_target[:,484:489].shape, training_target[:, 489].shape)
        tile_position_target = training_target[:, :484]
        move_direction_target = training_target[:,484:489]
        is50_target = training_target[:, 489].reshape(-1, 1)
        print(training_target.shape)
        print(training_target[0])
        model.fit(training_input,[tile_position_target, move_direction_target, is50_target], validation_split=0.1,
                  callbacks=[EarlyStopping(patience=10),
                             ModelCheckpoint(checkpoint_name,verbose=1,save_best_only=True),
                             tensorboard],
                  batch_size=BATCH_SIZE, nb_epoch=5, shuffle=True)
        
        epoch += 1
    #print('STILL accuracy:',model.evaluate(training_input[still_mask],training_target[still_mask],verbose=0)[1])
    #print('MOVE accuracy:',model.evaluate(training_input[~still_mask],training_target[~still_mask],verbose=0)[1])