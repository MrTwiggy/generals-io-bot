
import logging
import time
import math
import numpy as np
from generals_io_client import generals
from game import MAX_MAP_WIDTH, ORIGINAL_MAP_WIDTH, NUM_DIRECTIONS, NORTH, EAST, SOUTH, WEST
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
MAP_CHANNELS = 39


def pad(state, fill_value = 0, map_width = ORIGINAL_MAP_WIDTH):
    # Pad the frames to be 50x50
    x_diff = float(map_width - state.shape[1]) / 2
    x_padding = (math.ceil(x_diff), math.floor(x_diff))
    y_diff = float(map_width - state.shape[0]) / 2
    y_padding = (math.ceil(y_diff), math.floor(y_diff))
    #print("hhhh --", y_padding, x_padding, state.shape)
    return np.pad(state, pad_width=(y_padding, x_padding), 
                mode='constant', constant_values=fill_value), y_padding, x_padding

def generate_blank_state():
    return np.zeros((ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, MAP_CHANNELS)).astype('float32')
map_state = generate_blank_state()

def update_state(map_state, turn, tiles, armies, cities, generals_list, player, enemy, player_stats, enemy_stats, last_move = None):
    tiles, y_padding, x_padding = pad(np.array(tiles), MOUNTAIN)
    armies, y_padding, x_padding = pad(np.array(armies), 0)
    
    
    y_offset = y_padding[0]
    x_offset = x_padding[0]
    
    
    # Set state tiles with fog --- FOG FEATURE
    map_state[:,:,7] = (np.logical_or(tiles == FOG, tiles == OBSTACLE)).astype('float32')
    visible_tiles = map_state[:, :, 7] != 1
    
    # Sets whether a tile has ever been discovered --- DISCOVERED FEATURE
    map_state[:, :, 8] = np.logical_or(map_state[:, :, 8] == 1, map_state[:, :, 7] != 1).astype('float32')
    undiscovered_tiles = map_state[:, :, 8] != 1
    
    # Update number of turns in fog ---- TURNS IN FOG FEATURE
    map_state[:,:,9] += 1
    map_state[visible_tiles, 9] = 0
    
    # Set tile ownerships ----- OWNED BY AGENT FEATURE
    map_state[visible_tiles, 0] = tiles[visible_tiles] == player                            # Owned by us
    
    # ----- OWNED BY NEUTRAL FEATURE
    map_state[visible_tiles, 1] = tiles[visible_tiles] < 0 # Neutral
    map_state[undiscovered_tiles, 1] = 1 # Assume that all undiscovered tiles are neutral until discovered
    
    # ----- OWNED BY ENEMY FEATURE
    map_state[visible_tiles, 2] = tiles[visible_tiles] == enemy                          # Owned by enemy
    
    
    # ---- CITY TILE FEATURE
    for y, x in cities:
        map_state[y+y_offset, x+x_offset, 5] = 1              # Set cities
    
    # ---- GENERAL TILE FEATURE
    for y, x in generals_list:
        if y != -1 and x != -1:
            map_state[y+y_offset, x+x_offset, 6] = 1
    
    # Set tile types
    map_state[:, :, 3] = np.logical_or(tiles == EMPTY, tiles >= 0) # Set empty tiles
    map_state[:,:,3] = np.logical_or(map_state[:,:,3], tiles == FOG)
    map_state[:, :, 4] = np.logical_or(tiles == MOUNTAIN, tiles == OBSTACLE)# Set mountains
    
    # Ensure that known cities don't appear as other tile types
    city_tiles = map_state[:, :, 5] == 1
    map_state[city_tiles, 4] = 0 # Ensure that cities in fog don't get marked as mountains
    map_state[city_tiles, 3] = 0 # Ensure that cities that are owned aren't marked as empty
    
    #map_state[tiles != FOG, 9] = 0
    
    # Set army unit counts ---- ARMY TILE COUNT FEATURE
    map_state[visible_tiles, 10] = armies[visible_tiles]
    
    # Set whether this tile has enough army to validly be moved by us
    map_state[visible_tiles, 11] = np.logical_and(armies[visible_tiles] > 1, tiles[visible_tiles] == player)
    
    # Set whether this tile has enough army to validly be moved by enemy
    map_state[visible_tiles, 12] = np.logical_and(armies[visible_tiles] > 1, tiles[visible_tiles] == enemy)
    
    # Set current round in the game
    current_round = min(turn, 50*8 - 1) // 50
    round_features = np.arange(8) == current_round
    map_state[:, :, 13:21] = round_features
    
    # Set current time in round
    current_round_turn = (turn % 50) // 10
    turn_features = np.arange(5) == current_round_turn
    map_state[:, :, 21:26] = turn_features
    
    # Turns since last moved
    map_state[:, :, 32] += 1
    
    # Set last move made?
    if last_move is None:
        map_state[:, :, 26:31] = 0
    else:
        position, direction = last_move
        position = position.reshape(ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH)
        direction = direction.reshape(ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, 4)
        map_state[:, :, 26] = np.copy(position)
        map_state[:, :, 27:31] = np.copy(direction)
        map_state[position == 1, 32] = 0
    
    # Constant ones?
    map_state[:, :, 31] = 1
    
    # Enemy Army & Land
    map_state[:, :, 33] = enemy_stats[0]
    map_state[:, :, 34] = enemy_stats[1]
    
    # Player Army & Land
    map_state[:, :, 35] = player_stats[0]
    map_state[:, :, 36] = player_stats[1]
    
    # Whether player is currently in the lead or not
    map_state[:, :, 37] = enemy_stats[0] > player_stats[0]
    map_state[:, :, 38] = enemy_stats[1] > player_stats[1]
    
    # Constant zeros?
    #map_state[:, :, 27] = 0
    
    map_state = map_state.astype('float32')
    return map_state

def calculate_action(model, game_state, turn, tiles, armies, our_flag, enemy_flag, model_ver=0):
    if turn < 25:
        return None
    start_time = time.time()
    tiles_copy, y_padding, x_padding = pad(np.copy(tiles), MOUNTAIN, ORIGINAL_MAP_WIDTH)
    armies_copy, y_padding, x_padding = pad(np.copy(armies), 0, ORIGINAL_MAP_WIDTH)
    tile_mask = (np.logical_and(tiles_copy == our_flag, armies_copy > 1)).astype('float32')
    
    # Feed state into network and predict
    tile_position = np.zeros((ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH))
    move_direction = np.zeros((ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, NUM_DIRECTIONS))
    
    from train_imitation import augment_state, augment_direction
    num_dir = 4 if model_ver == 1 else 1
    num_flip = 2 if model_ver == 1 else 1
    for rotation in range(num_dir):
        for flip_vert in range(num_flip):
            # Augment the game state and then pass it through the policy network
            augmented_state = augment_state(np.copy(game_state), rotation, flip_vert)
            augmented_position, augmented_direction = model.predict(np.array([augmented_state]))
            augmented_position = augmented_position.reshape(ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH)
            augmented_direction = augmented_direction.reshape(ORIGINAL_MAP_WIDTH, ORIGINAL_MAP_WIDTH, NUM_DIRECTIONS)
            
            # De-augment the policy output back into it's original form and shape so it can be combined
            proper_position = augment_state(augmented_position, ((4 - rotation) % 4), flip_vert, flip_first = True)
            proper_direction = augment_direction(augmented_direction, ((4 - rotation) % 4), flip_vert, flip_first = True)
            
            # Combine this augmented policy with others
            tile_position += proper_position
            move_direction += proper_direction
    
    tile_position /= float(num_dir*num_flip)
    move_direction /= float(num_dir*num_flip)
    
    
    tile_position = tile_position * tile_mask
    tile_position = tile_position[y_padding[0]:ORIGINAL_MAP_WIDTH-y_padding[1], x_padding[0]:ORIGINAL_MAP_WIDTH-x_padding[1]]
    if np.sum(tile_position) <= 0:
        print("Skipping turn, no valid tile positions...")
        return None
    tile_position = tile_position / np.sum(tile_position)        
    tile_count = len(tile_position.flatten())        
        
    #-----TILE ACTION CHOICE------
    tile_index = np.random.choice(np.arange(tile_count), p=tile_position.flatten())
    #tile_index = np.argmax(tile_position.flatten())
    y, x = np.unravel_index(tile_index, tile_position.shape)
    x = int(x)
    y = int(y)
    
    #-----MOVE DIRECTION ACTION CHOICE------    
    cropped_state = game_state[y_padding[0]:ORIGINAL_MAP_WIDTH-y_padding[1], x_padding[0]:ORIGINAL_MAP_WIDTH-x_padding[1], :]
    height = cropped_state.shape[0]
    width = cropped_state.shape[1]
    move_direction = move_direction[y_padding[0]:ORIGINAL_MAP_WIDTH-y_padding[1], x_padding[0]:ORIGINAL_MAP_WIDTH-x_padding[1], :]
    move_direction = move_direction[y, x]
    
    # Check for mountains or map edges that will physically prevent us from moving there
    if y-1 <0 or cropped_state[y-1, x, 4] == 1:
        move_direction[0] = 0
    if x+1 >= width or cropped_state[y, x+1, 4] == 1:
        move_direction[1] = 0
    if y+1 >= height or cropped_state[y+1, x, 4] == 1:
        move_direction[2] = 0
    if x-1 <0 or cropped_state[y, x-1, 4] == 1:
        move_direction[3] = 0
    
    move_magnitude = np.sum(move_direction)
    if move_magnitude == 0:
        print("Skipping turn, no valid move directions...")
        return None
    move_direction /= move_magnitude
    target_move = np.argmax(move_direction)
    if move_direction[target_move] < 0.5:
        target_move = np.random.choice(np.arange(4), p=move_direction)
    
    #print("Positions: {}".format(tile_position))
    print("Tiles owned: {}/{}  by {}".format(np.sum(tile_mask), np.sum(tiles_copy == our_flag), our_flag))
    print("Move chosen: {} from {}".format(target_move, move_direction))
    if target_move == -1:
        y_dest = y
        x_dest = x
    elif target_move == NORTH:
        y_dest = y - 1
        x_dest = x
        move = "NORTH"
    elif target_move == EAST:
        y_dest = y
        x_dest = x + 1
        move = "EAST"
    elif target_move == SOUTH:
        y_dest = y + 1
        x_dest = x
        move = "SOUTH"
    elif target_move == WEST:
        y_dest = y
        x_dest = x - 1
        move = "WEST"
    
    print("Calculated action in {} seconds".format(time.time() - start_time))
    return y, x, y_dest, x_dest, y_padding, x_padding

def pretty_print(matrix):
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            sys.stdout.write(str(int(matrix[y, x])))
            sys.stdout.write(' ')
        sys.stdout.write('\n')
if __name__ == "__main__":
    from init_game import general
    import os, sys
    MODEL_NAME = sys.argv[1]
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
    last_move = None
    # ------------Main Bot Loop Logic Begins------------
    print("Waiting for updates...")
    for state in general.get_updates():
        print("STARTING MOVEE!!!")
        start_time = time.time()
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
        
        tiles_copy, y_padding, x_padding = pad(np.copy(tiles), MOUNTAIN, ORIGINAL_MAP_WIDTH)
        armies_copy, y_padding, x_padding = pad(np.copy(armies), 0, ORIGINAL_MAP_WIDTH)
        map_state = update_state(map_state, turn, tiles, armies, cities, generals_list, our_flag, enemy_flag, last_move)
        
        copy_state = np.copy(map_state).astype('int16')
        
        action = calculate_action(model, map_state, turn, tiles, armies, our_flag, enemy_flag)
        
        if action is None:
            print("No valid action this turn...")
            continue
        else:
            y, x, y_dest, x_dest, y_padding, x_padding = action
        
        move = " "
        x_dest = int(x_dest)
        y_dest = int(y_dest)
        print("Submitting move {} from ({}, {}) to ({}, {}), computed in {} seconds".format(move, y, x, y_dest, x_dest, time.time() - start_time))
        
        # Keep track of the last submitted move
        from LoadReplayData import coordinates_to_direction, generate_target_tensors
        direction = coordinates_to_direction(y, x, y_dest, x_dest)
        last_move = generate_target_tensors(y, x, direction)
        
        # Submit the calculate move
        general.move(y, x, y_dest, x_dest)








