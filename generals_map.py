import numpy as np

class Map:
    TILE_EMPTY        = -1
    TILE_MOUNTAIN     = -2
    TILE_FOG          = -3
    TILE_FOG_OBSTACLE = -4
    
    def __init__(self, width, height, teams = None):
        self.width = width
        self.height = height
        self.teams = teams

        self._map = np.full(self.width * self.height, self.TILE_EMPTY)
        self._armies = np.zeros(self.width * self.height)

    def size(self):
        return self.width * self.height

    def index_for(self, row, col):
        return row * self.width + col
        
    def has_view_of(self, player, tile):
        tile_y = tile // self.width
        tile_x = tile % self.width
        
        for x_diff in [-1, 0, 1]:
            for y_diff in [-1, 0, 1]:
                ind = self.index_for(tile_y + y_diff, tile_x + x_diff)
                if self.is_valid_tile(ind) and self.tile_at(ind) == player:
                    return True
        
        return False
        
    def is_adjacent(self, ind1, ind2):
        return self.distance(ind1, ind2) == 1

    def is_valid_tile(self, ind):
        return ind >= 0 and ind < len(self._map)
    
    def is_owned(self, ind):
        return self.tile_at(ind) >= 0
    
    def tile_at(self, ind):
        return self._map[ind]

    def army_at(self, ind):
        return self._armies[ind]

    def set_tile(self, ind, val):
        self._map[ind] = val
        
    def set_army(self, ind, val):
        self._armies[ind] = val

    def increment_army_at(self, ind):
        self._armies[ind] += 1

    def attack(self, start, end, is50, generals):
        if not self.is_valid_tile(start):
            print("c")
            return False
        
        if not self.is_valid_tile(end):
            print("e")
            return False

        if not self.is_adjacent(start, end):
            print("g")
            return False

        if self.tile_at(end) == Map.TILE_MOUNTAIN:
            print("ff")
            return False
        
        reserve = int(np.ceil(self._armies[start] / 2)) if is50 else 1

        
        # Attacking an enemy
        if not self.teams or \
           self.teams[self.tile_at(start)] != self.teams[self.tile_at(end)]:
            
            if self._armies[start] <= 1:
                # If the army at the start tile is <= 1, the attack fails.
                return False
            elif self.tile_at(end) == self.tile_at(start):
                # self -> self
                self._armies[end] += self._armies[start] - reserve
            else:
                # self -> enemy
                if self._armies[end] >= self._armies[start] - reserve:
                    # Non-takeover
                    self._armies[end] -= self._armies[start] - reserve
                else:
                    # Takeover
                    self._armies[end] = self._armies[start] - reserve - self._armies[end]
                    self.set_tile(end, self.tile_at(start))

        # Attacking an ally
        else:
            self._armies[end] += self._armies[start] - reserve

            # Take ownership of army only if not a general tile
            if not np.where(generals == end)[0]:
                self.set_tile(end, self.tile_at(start))

        self._armies[start] = reserve

        return True

    def replace_all(self, val1, val2, scale = 1):
        for i in range(len(self._map)):
            if self._map[i] == val1:
                self._map[i] = val2
                self._armies[i] *= scale

    def distance(self, ind1, ind2):
        r1 = ind1 // self.width
        c1 = ind1 % self.width
        r2 = ind2 // self.width
        c2 = ind2 % self.width

        return abs(r1 - r2) + abs(c1 - c2)

                    
                
