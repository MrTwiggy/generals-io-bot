import numpy as np
import generals_map

from generals_map import Map

class Game:
    DEAD_GENERAL = -1
    
    @classmethod
    def from_replay(replay):
        width       = replay["mapWidth"]
        height      = replay["mapHeight"]
        cities      = replay["cities"]
        city_armies = replay["cityArmies"]
        generals    = replay["generals"]
        mountains   = replay["mountains"]
        moves       = replay["moves"]
        players     = replay["usernames"]
        teams       = replay["teams"]
        
        game = new Game(players, teams)
        
        game.cities   = []
        game.generals = []
        
        game.gmap = new Map(width, height, teams)

        # Replicate the initial state
        for ind in mountains:
            game.add_mountain(ind)
            
        for ind, size in zip(cities, city_armies):
            game.add_city(ind, size)

        for ind in generals:
            game.add_general(ind)
    
    
    def __init__(self, usernames, teams):
        self.usernames = usernames
        self.teams     = teams

        self.num_players   = len(usernames)
        self.turn          = 0
        self.alive_players = len(usernames)
        self.input_buffer  = []
        self.tile_count    = np.ones(num_players)
        self.army_count    = np.ones(num_players)
        self.deaths        = np.full(num_players, -1)
        
        
    def add_mountain(self, ind):
        self.gmap.set_tile(ind, Map.TILE_MOUNTAIN)

    def add_city(self, ind, army):
        self.gmap.set_army(ind, army)
        
    def add_general(self, ind):
        self.generals.append(ind)
        self.gmap.set_tile(ind, len(self.generals) - 1)
        self.gmap.set_army(ind, 1)


    def is_over(self):
        if not self.teams and self.alive_players == 1:
            return True
        else:
            winning_team = None
            return False
                
            
    def handle_attack(self, player, start, end, is50, attack_ind):
        if self.gmap.tile_at(start) != player:
            return False

        end_tile = self.gmap.tile_at(end)

        succeeded = self.gmap.attack(start, end, is50, self.generals)

        if not succeeded:
            return False

        new_end_tile = self.gmap.tile_at(end)
        if new_end_tile != end_tile and np.where(self.generals == end)[0]:
            general_ind = np.where(self.generals == end)[0][0]

            # general captured! Give the capturer command of the captured's army
            self.gmap.replace_all(end_tile, new_end_tile, 0.5)

            self.deaths.append(general_ind)
            self.alive_players -= 1
            
            # Turn the general into a city
            self.cities.push(end)
            self.generals[general_ind] = DEAD_GENERAL

    def alive_teammate(self, ind):
        return None

    
