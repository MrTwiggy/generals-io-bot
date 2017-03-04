
import logging
import math
import numpy as np
from generals_io_client import generals
from game import MAX_MAP_WIDTH
import sys
logging.basicConfig(level=logging.DEBUG)


def position_plus(a,b):
    return (a[0]+b[0],a[1]+b[1])

def grid_to_index(y,x):
    return y*cols+x

def index_to_grid(index):
    return index/cols, index%cols


#GENERAL = 0
EMPTY = -1
MOUNTAIN = -2
FOG = -3
OBSTACLE = -4
MAP_CHANNELS = 11
MODEL_NAME = sys.argv[1]

def pad(state, fill_value = 0, map_width = MAX_MAP_WIDTH):
    # Pad the frames to be 50x50
    x_diff = float(map_width - state.shape[1]) / 2
    x_padding = (math.ceil(x_diff), math.floor(x_diff))
    y_diff = float(map_width - state.shape[0]) / 2
    y_padding = (math.ceil(y_diff), math.floor(y_diff))
    #print("hhhh --", y_padding, x_padding, state.shape)
    return np.pad(state, pad_width=(y_padding, x_padding), 
                mode='constant', constant_values=fill_value), y_padding, x_padding

def generate_blank_state():
    return np.zeros((MAX_MAP_WIDTH, MAX_MAP_WIDTH, MAP_CHANNELS)).astype('int32')
map_state = generate_blank_state()

def update_state(map_state, tiles, armies, cities, generals_list, player, enemy):
    tiles, y_padding, x_padding = pad(np.array(tiles), MOUNTAIN)
    armies, y_padding, x_padding = pad(np.array(armies), 0)
    
    
    y_offset = y_padding[0]
    x_offset = x_padding[0]
    # Set tile ownerships
    map_state[:,:,0] = tiles == player                            # Owned by us
    map_state[:,:,1] = tiles < 0 # Neutral
    map_state[:,:,2] = tiles == enemy                          # Owned by enemy
    
    # Set tile types
    map_state[:, :, 3] = tiles == EMPTY # Set empty tiles
    map_state[:, :, 4] = tiles == MOUNTAIN# Set mountains
    
    for y, x in cities:
        map_state[y+y_offset, x+x_offset, 5] = 1              # Set cities
    
    for y, x in generals_list:
        if y != -1 and x != -1:
            map_state[y+y_offset, x+x_offset, 6] = 1
    
    # Set state tiles with fog
    map_state[:,:,7] = (np.logical_or(tiles == FOG, tiles == OBSTACLE)).astype('int32')
    
    # Sets whether a tile has ever been discovered
    map_state[:, :, 8] = np.logical_or(map_state[:, :, 8] == 1, map_state[:, :, 7] != 1)
    
    # Update number of turns in fog
    map_state[:,:,9] += 1
    map_state[np.logical_and(tiles != FOG, tiles != OBSTACLE), 9] = 0
    #map_state[tiles != FOG, 9] = 0
    
    # Set army unit counts
    map_state[:,:,10] = armies
    return map_state



if __name__ == "__main__":
    from init_game import general
    import os, sys
    model = None
    #from keras.models import load_model
    print("---Loading model----")
    #model = load_model('{}.h5'.format(MODEL_NAME), {'multi_label_crossentropy' : multi_label_crossentropy})
    from train_imitation import load_model_train
    model = load_model_train("./data", MODEL_NAME)
    print("---Loaded model!----")
    # first_update=general.get_updates()[0]
    # rows=first_update['rows']
    # cols=first_update['cols']
    # our_flag=first_update['player_index']
    # general_y, general_x =first_update['generals'][our_flag]
    our_flag=0
    
    general_y, general_x =0,0
    general_position=(general_y,general_x)
    tiles=[]
    armies=[]
    cities=[]
    generals_list=[]
    # ------------Main Bot Loop Logic Begins------------
    print("Waiting for updates...")
    for state in general.get_updates():
        print("STARTING MOVEE!!!")
        # get position of your general
        our_flag = state['player_index']
        enemy_flag = 1 if our_flag == 0 else 0
        try:
            general_y, general_x = state['generals'][our_flag]
        except KeyError:
            break
    
        rows, cols = state['rows'], state['cols']
    
        turn = state['turn']
        tiles = np.array(state['tile_grid'])
        armies = np.array(state['army_grid'])
        cities = state['cities']
        generals_list = state['generals']
        
        tiles_copy, y_padding, x_padding = pad(np.copy(tiles), MOUNTAIN, 22)
        armies_copy, y_padding, x_padding = pad(np.copy(armies), 0, 22)
        # Update the map_state which is a 22x22x11 map tile array with 11 channels per tile
        update_state(map_state, tiles, armies, cities, generals_list, our_flag, enemy_flag)
        #map_state[:, :, 0:3] (Owned by me, Neutral, Owned by enemy)
        #map_state[:, :, 3:7] (Normal, Mountain, City, General)
        #map_state[:, :, 7] Tile is in fog
        #map_state[:, :, 8] Tile has been discoevered
        #map_state[:, :, 9] NUmber of turns in fog
        #map_state[:, :, 10] Number of army units on tile
        
        model_output = model.predict(np.array([map_state]))
        
        tile_position, move_direction, is50 = model_output[0], model_output[1], model_output[2]
        tile_position = tile_position.reshape(22, 22)
        
        tile_mask = (tiles_copy == our_flag).astype('float32')
        print(tile_position.astype('float32'))
        print(move_direction)
        print(move_direction.argmax())
        print(is50)
        tile_position = tile_position * tile_mask
        
        tile_position = tile_position[y_padding[0]:22-y_padding[1], x_padding[0]:22-x_padding[1]]
        y, x = np.unravel_index(tile_position.argmax(), tile_position.shape)
        print("Best pos: ", y, x)
        print("General: ", general_y, general_x)
        target_move = move_direction.argmax()
        STILL = 0
        NORTH = 1
        EAST = 2
        SOUTH = 3
        WEST = 4
        x = int(x)
        y = int(y)
        
        if target_move == STILL:
            y_dest = y
            x_dest = x
        elif target_move == NORTH:
            y_dest = y - 1
            x_dest = x
        elif target_move == EAST:
            y_dest = y
            x_dest = x + 1
        elif target_move == SOUTH:
            y_dest = y + 1
            x_dest = x
        elif target_move == WEST:
            y_dest = y
            x_dest = x - 1
        
        print(state['replay_url'])
        x_dest = int(x_dest)
        y_dest = int(y_dest)
        print("Submitting move from ({}, {}) to ({}, {})".format(y, x, y_dest, x_dest))
        general.move(y, x, y_dest, x_dest)
        # TODO: Feed the above state into neural network, get output move, and then
        # submit move using generals.move(y_origin, x_origin, y_destination, x_destination) and then repeat








