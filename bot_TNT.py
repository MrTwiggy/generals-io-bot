from init_game import general
import logging
from generals_io_client import generals
import math
import numpy as np

logging.basicConfig(level=logging.DEBUG)

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
MAX_MAP_WIDTH = 22

def pad(state, fill_value = 0):
    # Pad the frames to be 50x50
    x_diff = float(MAX_MAP_WIDTH - state.shape[1]) / 2
    x_padding = (math.ceil(x_diff), math.floor(x_diff))
    y_diff = float(MAX_MAP_WIDTH - state.shape[0]) / 2
    y_padding = (math.ceil(y_diff), math.floor(y_diff))
    #print("hhhh --", y_padding, x_padding, state.shape)
    return np.pad(state, pad_width=(y_padding, x_padding), 
                mode='constant', constant_values=fill_value), y_padding, x_padding

map_state = np.zeros((MAX_MAP_WIDTH, MAX_MAP_WIDTH, 11)).astype('int32')

def update_state(tiles, armies, cities, generals_list):
    tiles, y_padding, x_padding = pad(np.array(tiles), MOUNTAIN)
    armies, y_padding, x_padding = pad(np.array(armies), 0)
    
    
    y_offset = y_padding[0]
    x_offset = x_padding[0]
    # Set tile ownerships
    map_state[:,:,0] = tiles == our_flag                            # Owned by us
    map_state[:,:,1] = tiles < 0 # Neutral
    map_state[:,:,2] = tiles == enemy_flag                          # Owned by enemy
    
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



# ------------Main Bot Loop Logic Begins------------
for state in general.get_updates():

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
    
    # Update the map_state which is a 22x22x11 map tile array with 11 channels per tile
    update_state(tiles, armies, cities, generals_list)
    #map_state[:, :, 0:3] (Owned by me, Neutral, Owned by enemy)
    #map_state[:, :, 3:7] (Normal, Mountain, City, General)
    #map_state[:, :, 7] Tile is in fog
    #map_state[:, :, 8] Tile has been discoevered
    #map_state[:, :, 9] NUmber of turns in fog
    #map_state[:, :, 10] Number of army units on tile
    
    # TODO: Feed the above state into neural network, get output move, and then
    # submit move using generals.move(y_origin, x_origin, y_destination, x_destination) and then repeat








