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

from LoadReplayData import load_all_replays, generate_target_tensors, sample_dataset, get_dataset_info

from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import clip_ops

MAP_CHANNELS = 39

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
    move_counts = tf_count(K.cast(move_mask, dtype='int32'), 1)
    
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


def generate_paddings(height, width):
    diff_x = float(ORIGINAL_MAP_WIDTH - width) / 2.0
    diff_y = float(ORIGINAL_MAP_WIDTH - height) / 2.0
    
    return (math.ceil(diff_y), math.floor(diff_y)), (math.ceil(diff_x), math.floor(diff_x))

def augment_state(original_state, rotation=0, flip_vert=0, flip_first=False):
    state = np.copy(original_state)
    
    
    if flip_vert and flip_first:
        state = np.flipud(state)
        
    state = np.rot90(state, k=rotation, axes=(1,0))
    
    if flip_vert and not flip_first:
        state = np.flipud(state)
        
    return state

def augment_direction(original_direction, rotation=0, flip_vert=0, flip_first=False):
    move_direction = np.copy(original_direction)
    
    
    if flip_vert and flip_first:
        move_direction = np.flipud(move_direction)
        temp = np.copy(move_direction[:, :, 0])
        move_direction[:, :, 0] = move_direction[:, :, 2]
        move_direction[:, :, 2] = temp
    
    move_direction = np.rot90(move_direction, k=rotation, axes=(1,0))
    move_copy = np.copy(move_direction)
    for direction in range(4):
        prev_direction = (direction - rotation) % 4
        move_direction[:, :, direction] = move_copy[:, :, prev_direction]
    
    if flip_vert and not flip_first:
        move_direction = np.flipud(move_direction)
        temp = np.copy(move_direction[:, :, 0])
        move_direction[:, :, 0] = move_direction[:, :, 2]
        move_direction[:, :, 2] = temp 
    
    return move_direction

def augment_batch(batch_X, batch_y, batch_size, uniform_rotation=None, uniform_flip = None):
    for i in range(batch_size):
        rotation = uniform_rotation if uniform_rotation is not None else randint(0, 3)
        flip_vert = uniform_flip if uniform_flip is not None else randint(0,1)
        
        tile_length = ORIGINAL_MAP_WIDTH**2
        example_tile = np.copy(batch_y[i, :tile_length]).reshape(ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH)
        example_direction = np.copy(batch_y[i, tile_length:5*tile_length]).reshape(ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, 4)
        #example_oracle_state = np.copy(batch_y[i, 5*tile_length:]).reshape(ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, 8)
        #direction = np.argmax(example_direction[example_tile == 1])
        
        # Augment the game state and tile target and move direction target
        last_move_direction = np.copy(batch_X[i][:, :, 27:31])
        batch_X[i] = augment_state(batch_X[i], rotation, flip_vert)
        batch_X[i][:, :, 27:31] = augment_direction(last_move_direction, rotation, flip_vert)
        #final_oracle_state = augment_state(example_oracle_state, rotation, flip_vert)
        example_tile = augment_state(example_tile, rotation, flip_vert)
        final_direction = augment_direction(example_direction, rotation, flip_vert)
        
        batch_y[i, :tile_length] = example_tile.flatten()
        batch_y[i, tile_length:5*tile_length] = final_direction.flatten()
        #batch_y[i, 5*tile_length:] = final_oracle_state.flatten()
        
    
    batch_X = np.concatenate(np.array([batch_X]), axis=0)
    batch_y = np.concatenate(np.array([batch_y]), axis=0)
    
    for i in range(len(batch_y)):
        batch_y[i] = batch_y[i].astype('float32')
    batch_y = np.array([batch_y[i] for i in range(len(batch_y))], dtype='float32')
    batch_y = batch_y.reshape(batch_size, -1)
    
    for i in range(len(batch_y)):
        batch_X[i] = batch_X[i].astype('float32')
    batch_X = np.array([batch_X[i] for i in range(len(batch_X))], dtype='float32')
    
    batch_X = batch_X.reshape(batch_size, ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, MAP_CHANNELS)
    
    return batch_X, batch_y

class BatchGenerator:
    def __init__(self, file_name, dataset_name, batch_size, discount=1.0, augment=False,workers=1):
        self.generators = [frame_generator(file_name, dataset_name, batch_size, discount, augment) for i in range(workers)]
        self.generator_index = 0
        self.lock = multiprocessing.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
           index = self.generator_index
           generator = self.generators[index]
           self.generator_index = (self.generator_index + 1) % len(self.generators)
           #print(self.generator_index, len(self.generators))
        return next(generator)

def generate_batch(batch_X, batch_y, augment, completed_batches, discount=0.999, version=0, rotation=None, flip=None):
    batch_X, batch_y = np.copy(batch_X), np.copy(batch_y)
    #print(batch_X.dtype, batch_y.dtype)
    current_batch_size = len(batch_X)
    # Construct the target tensors
    start = time.time()
    tile_target = []
    move_direction_target = []
    game_outcome_target = []
    oracle_state_target = []
    for i in range(current_batch_size):
        y, x, direction, height, width, winner, turns_left = batch_y[i, 0:7]
        tile_pos, move_direction = generate_target_tensors(x, y, direction)
        tile_target.append(tile_pos.flatten())
        move_direction_target.append(move_direction.flatten())
        
        
        #print(y, x, direction, height, width, winner, turns_left)
        game_outcome = (discount**turns_left) * winner
        game_outcome_target.append(game_outcome)
        #oracle_state = np.copy(batch_y[i, 7:]).flatten()
        #game_outcome_target.append(game_outcome)
        #oracle_state_target.append(oracle_state)
   
    tile_target = np.array(tile_target).astype('float32')
    move_direction_target = np.array(move_direction_target).astype('float32')
    game_outcome_target = np.array(game_outcome_target).astype('float32').reshape(len(tile_target), 1)
    #print("Winners: ", game_outcome_target)
    #print(game_outcome_target.shape, tile_target.shape, move_direction_target.shape)
    #oracle_state_target = np.array(oracle_state_target).astype('float32')
    batch_y = np.concatenate((tile_target, move_direction_target), axis=1)#, oracle_state_target), axis=1)
    set_duration = time.time() - start
    start_augment = time.time()
    
    # Perform random data augmentation on frames independantly
    if augment:
        batch_X, batch_y = augment_batch(batch_X, batch_y, current_batch_size, rotation, flip)
    
    augment_duration = time.time() - start_augment
    # Yield the augmentated and formatted example batch that was loaded off disk
    tile_len = tile_target.shape[1]
    if completed_batches % 100 == 0:
        tf.reset_default_graph()
    
    final_batch_y = [batch_y[:, :tile_len], batch_y[:, tile_len:5*tile_len]]
    
    if version == 1:
        final_batch_y = [np.concatenate((batch_y[:, :tile_len], game_outcome_target), axis=1), np.concatenate((batch_y[:, tile_len:5*tile_len], game_outcome_target), axis=1)]
    return batch_X, final_batch_y
# training_input_orig is of shape (num_samples, map_height, map_width, 11)
# training_target_orig is of shape (num_samples, 1),
# each row in target is a tuple of ((y,x), direction) where
# x,y are the positions of the moving tile and direction is an int in [0,3] denoting the direction moved
def frame_generator(file_name, dataset_name, batch_size, discount=1.0, augment=False):
    while True:
        input_shape, target_shape = get_dataset_info(file_name, dataset_name)
        sample_count = input_shape[0]
        batches = math.ceil(float(sample_count) / batch_size)
        print("Beginning frame generation loop with {} samples available for {} total batches!".format(sample_count, batches))
       
        indices = np.arange(sample_count)
        np.random.shuffle(indices)
       
        batch_index_sets = np.split(indices, [batch_size*i for i in range(1, batches)])
        print("Batch index sets: ", len(batch_index_sets))
        completed_batches = 0
        for batch_indices in batch_index_sets:
            # Load and sample the data off disk from dataset h5 file
            start_time = time.time()
            current_batch_size = len(batch_indices)
            #print(sorted(batch_indices))
            start = time.time()
            batch_X, batch_y = sample_dataset(file_name, dataset_name, current_batch_size, sorted(batch_indices))
            read_duration = time.time() - start
           
           
            batch_X, batch_y = generate_batch(batch_X, batch_y, augment, completed_batches)
            completed_batches += 1
            yield batch_X, batch_y

def build_model():
    #---- Shared convolution network 
    main_input = Input(shape=(ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, MAP_CHANNELS,), dtype='float32', name='main_input')
    
    n_filters = 196
    n_shared_layers = 10
    shared_layers = []

    cur_layer = Convolution2D(n_filters, 5, 5, border_mode="same",activation = 'relu')(main_input)
    cur_layer = BatchNormalization()(cur_layer)
    shared_layers.append(cur_layer)

    for i in range(1, n_shared_layers):
        cur_layer = Convolution2D(n_filters, 3, 3, border_mode="same", activation = 'relu')(cur_layer)
        cur_layer = BatchNormalization()(cur_layer)
        shared_layers.append(cur_layer)
    
    #---- Tile position network
    tile_position_layers = []
    tile_position = cur_layer
    for i in range(5):
        tile_position = Convolution2D(n_filters, 3, 3, border_mode="same", activation = 'relu')(tile_position)
        tile_position = BatchNormalization()(tile_position)
        tile_position_layers.append(tile_position)
    
    
    tile_position = keras.layers.Merge(mode='concat')(shared_layers + tile_position_layers)
    tile_position = Convolution2D(1, 1, 1, border_mode="same", activation='linear')(tile_position)
    tile_position = Flatten()(tile_position)
    tile_position = Activation('softmax', name='tile_position')(tile_position)  
    
    #------ Move direction network
    move_direction_layers = []
    move_direction = cur_layer
    for i in range(5):
        move_direction = Convolution2D(n_filters, 3, 3, border_mode="same", activation = 'relu')(move_direction)
        move_direction = BatchNormalization()(move_direction)
        move_direction_layers.append(move_direction)
    
    move_direction = keras.layers.Merge(mode='concat')(shared_layers + move_direction_layers)
    move_direction = Convolution2D(4, 1, 1, border_mode="same", activation='linear')(move_direction)
    move_direction = Reshape((ORIGINAL_MAP_WIDTH*ORIGINAL_MAP_WIDTH, 4))(move_direction)
    move_direction = Activation('softmax')(move_direction)  
    move_direction = Flatten(name='move_direction')(move_direction)    
    
    #------ Value Network Outcome Prediction
    """game_outcome = Convolution2D(64, 3, 3, border_mode="same", activation = 'relu')(cnn)
    game_outcome = BatchNormalization()(game_outcome)
    game_outcome = Flatten()(game_outcome)
    #game_outcome = Dense(256, activation='relu')(game_outcome)
    game_outcome = Dense(1, activation='tanh', name='game_outcome')(game_outcome)"""
    
    #------ Oracle map state prediction from observation
    """oracle_state = Convolution2D(64, 3, 3, border_mode="same", activation = 'relu')(cnn)
    oracle_state = BatchNormalization()(oracle_state)
    oracle_state = Convolution2D(8, 9, 9, border_mode="same", activation='linear')(oracle_state)
    oracle_state = Flatten(name='oracle_state')(oracle_state)    """
    
    #----- Is half-move network
    #is50 = Dense(1, activation='sigmoid', name='is50')(merge([flattened_cnn, tile_position, move_direction], mode='concat'))
    
    #---- Final built model
    model = Model(input=main_input, output=[tile_position, move_direction])#, game_outcome, oracle_state])#, is50])
    model_loss = {'tile_position': 'categorical_crossentropy', 'move_direction': multi_label_crossentropy, 
                  }
    model_metrics = {'move_direction' : multi_label_accuracy, 'tile_position':'accuracy'}
    model_weighting = {'tile_position': 1.0, 'move_direction' : 1.0}
    model.compile(keras.optimizers.Adam(lr=0.0006, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss=model_loss, metrics=model_metrics, loss_weights=model_weighting)
    
    return model

def load_model_train(dataFolder, modelName, custom_objs=None):
    model_path = '{}/{}.h5'.format(dataFolder, modelName)
    if not os.path.exists(model_path):
        return None
    
    #model = build_model()
    #model.load_weights(model_path)
    if custom_objs is None:
        #custom_objs = {'multi_label_crossentropy':multi_label_crossentropy, 'multi_label_accuracy':multi_label_accuracy}
        custom_objs = {'tile_position_loss':tile_position_loss, 'move_direction_loss':move_direction_loss, 'multi_label_crossentropy':multi_label_crossentropy, 'multi_label_accuracy':multi_label_accuracy}
    model = keras.models.load_model(model_path, custom_objects=custom_objs)
    return model

# --------------------- Main Training Logic -----------------------
if __name__ == "__main__":
    arg_count = len(sys.argv) - 1
    
    MODEL_NAME = sys.argv[1] if arg_count >= 1 else "default-model"
    DATA_FILE_NAME = sys.argv[2] if arg_count >= 2 else "default-data"
    GPUS = sys.argv[3] if arg_count >= 3 else "0"
    BATCH_SIZE = int(sys.argv[4]) if arg_count >= 4 else 64
    os.environ["CUDA_VISIBLE_DEVICES"]=GPUS
    DISCOUNT = 1.0
    DATA_FOLDER = "./data"
    
    print("----STARTING TRAINING------")
    #np.random.seed(1337) # for reproducibility  
    
    model = load_model_train(DATA_FOLDER, MODEL_NAME)
    if model is None:
        model = build_model()
        #model.load_weights("./data/{}.h5".format(MODEL_NAME))
    
    for epoch in range(1,30):
        print("Initializing meta epoch: ", epoch)
        print(model.summary())
        checkpoint_name = '{}/{}v{}.h5'.format(DATA_FOLDER, MODEL_NAME, epoch)
        print("Checkpoint name: ", checkpoint_name)
        now = datetime.datetime.now()
        tensorboard = TensorBoard(log_dir='./logs/'+now.strftime('%Y.%m.%d %H.%M'))
        
        training_input_shape, training_target_shape = get_dataset_info(DATA_FILE_NAME, "training")
        validation_input_shape, validation_target_shape = get_dataset_info(DATA_FILE_NAME, "validation")
        print("Training shapes: ", training_target_shape, training_input_shape)
        print("Validation shapes: ", validation_target_shape, validation_input_shape)
        
        model.fit_generator(
            BatchGenerator(DATA_FILE_NAME, "training", BATCH_SIZE, discount=DISCOUNT, augment=True, workers=1),
            validation_data=BatchGenerator(DATA_FILE_NAME, "validation", BATCH_SIZE, discount=DISCOUNT, augment=False, workers=1),
            samples_per_epoch=training_input_shape[0]/4,
            nb_val_samples=validation_input_shape[0]/4,
            nb_epoch=12,
            verbose=1, max_q_size=25, #workers=1,
            callbacks=[EarlyStopping(patience=8), tensorboard,
                       ModelCheckpoint(checkpoint_name,verbose=1,save_best_only=True)])
        
        
        
