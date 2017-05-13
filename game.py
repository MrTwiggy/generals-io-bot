import numpy as np
import generals_map
import scipy.ndimage as ndimage
from generals_map import Map

MAX_MAP_WIDTH = 31
ORIGINAL_MAP_WIDTH = 23
NUM_DIRECTIONS = 4
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

class Game:
    DEAD_GENERAL = -1
    RECRUIT_RATE = 2 # How often generators generate a unit (in turns)
    FARM_RATE = 50  # How often owned tiles generate a unit (in turns)
    MIN_CITY_ARMY = 40 # MInimum number of units in a city before it begins generating
    DILATION_STRUCTURE = np.ones(9).reshape(3,3)
    
    @staticmethod
    def from_replay(replay, version=7):
        width       = replay["mapWidth"]
        height      = replay["mapHeight"]
        cities      = replay["cities"]
        city_armies = replay["cityArmies"]
        generals    = replay["generals"]
        mountains   = replay["mountains"]
        moves       = replay["moves"]
        players     = replay["usernames"]
        teams       = replay["teams"]
        
        game = Game(players, teams, version)
        
        game.gmap = Map(width, height, teams)

        # Replicate the initial state
        for ind in mountains:
            game.add_mountain(ind)
            
        for ind, size in zip(cities, city_armies):
            game.add_city(ind, size)

        for ind in generals:
            game.add_general(ind)
        
        return game
    
    
    def __init__(self, usernames, teams, version=7):
        self.usernames = usernames
        self.teams     = teams
        self.version = version
        
        self.cities = []
        self.generals = []
        self.num_players   = len(usernames)
        self.turn          = 1
        self.alive_players = len(usernames)
        self.input_buffer  = []
        self.tile_count    = np.ones(self.num_players)
        self.army_count    = np.ones(self.num_players)
        self.deaths        = []
        
    
    def generate_state(self, player, oracle = False):
        tiles = np.copy(self.gmap._map).reshape(self.gmap.height, self.gmap.width)
        armies = np.copy(self.gmap._armies).reshape(self.gmap.height, self.gmap.width)
        cities = []
        generals = []
        
        # Add generals that can be seen by the player
        for general in self.generals:
            if self.gmap.has_view_of(player, general) or oracle:
                generals.append(self.index_to_coordinates(general))
        
        # Calculate a masking of visible territory on the map
        visible_tiles = (np.copy(tiles) == player).astype('int8')
        visible_tiles = ndimage.binary_dilation(visible_tiles, structure=Game.DILATION_STRUCTURE)
        
        if not oracle:
            # Reduce all unseen tiles' army counts to zero to prevent impossible information.
            armies *= visible_tiles
            
            # Set the fog of war tiles so that no unfair information is given to the player
            tiles[np.logical_and(visible_tiles == 0, tiles != Map.TILE_MOUNTAIN)] = Map.TILE_FOG
            tiles[np.logical_and(visible_tiles == 0, tiles == Map.TILE_MOUNTAIN)] = Map.TILE_FOG_OBSTACLE
        
        #Add cities that can be seen by the player
        for city in self.cities:
            if self.gmap.has_view_of(player, city) or oracle:
                cities.append(self.index_to_coordinates(city))
            
            if not self.gmap.has_view_of(player, city) and not oracle:
                y, x = self.index_to_coordinates(city)
                tiles[y, x] = Map.TILE_FOG_OBSTACLE
        return tiles.flatten(), armies.flatten(), cities, generals
    
    def winner(self):
        for i in range(len(self.usernames)):
            if i not in self.deaths:
                return i
        return None
    
    def index_to_coordinates(self, ind):
        return ind // self.gmap.width, ind % self.gmap.width
    
    def add_mountain(self, ind):
        self.gmap.set_tile(ind, Map.TILE_MOUNTAIN)

    def add_city(self, ind, army):
        self.cities.append(ind)
        self.gmap.set_army(ind, army)
        
    def add_general(self, ind):
        self.generals.append(ind)
        self.gmap.set_tile(ind, len(self.generals) - 1)
        self.gmap.set_army(ind, 1)
    
    def tiles_owned(self, player):
        return np.sum(self.gmap._map == player)
    def end_game(self):
        if self.tiles_owned(0) > self.tiles_owned(1):
            self.kill_player(1)
        else:
            self.kill_player(0)
    def is_over(self):
        if not self.teams and self.alive_players == 1:
            return True
        else:
            return False
                
    def print_game(self):
        print("--------TURN {}---------".format(self.turn))
        print(np.array(self.gmap._armies, dtype='int16').reshape(self.gmap.height, self.gmap.width))
        print(np.array(self.gmap._map, dtype='int16').reshape(self.gmap.height, self.gmap.width))
    
    def kill_player(self, player):
        self.alive_players -= 1
        self.deaths.append(player)
    
    def handle_attack(self, player, start, end, is50):
        if not self.gmap.in_bounds(start) or not self.gmap.in_bounds(end):
            print("Start or end tile was out of map bounds! Rejecting move...")
            return False
        if self.gmap.tile_at(start) != player:
            #print("a")
            return False

        end_tile = self.gmap.tile_at(end)

        succeeded = self.gmap.attack(start, end, is50, self.generals)

        if not succeeded:
            #print("b")
            return False

        new_end_tile = self.gmap.tile_at(end)
        if new_end_tile != end_tile and end in self.generals:
            killed_player = self.generals.index(end)

            # general captured! Give the capturer command of the captured's army
            self.gmap.replace_all(end_tile, new_end_tile, 0.5)

            self.kill_player(killed_player)
            
            # Turn the general into a city
            self.cities.append(end)
            self.generals[killed_player] = Game.DEAD_GENERAL
            print("GENERAL CAPTURED BY ", new_end_tile)
        return True

    def alive_teammate(self, ind):
        return None
    
    def update(self):
        self.turn += 1
        
        if (self.turn % Game.RECRUIT_RATE == 0):
            for general_ind in self.generals:
                self.gmap.increment_army_at(general_ind)
            
            for city_ind in self.cities:
                if self.gmap.is_owned(city_ind) or (self.gmap.army_at(city_ind) < Game.MIN_CITY_ARMY and self.version < 7):
                    self.gmap.increment_army_at(city_ind)
        
        if (self.turn % Game.FARM_RATE == 0):
            for i in range(self.gmap.size()):
                if self.gmap.is_owned(i):
                    self.gmap.increment_army_at(i)
            

    
