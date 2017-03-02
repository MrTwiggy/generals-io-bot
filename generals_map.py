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

    def is_adjacent(self, ind1, ind2):
        r1 = ind1 / self.width
        c1 = ind1 % self.width
        r2 = ind2 / self.width
        c2 = ind2 % self.width

        return abs(r1 - r2) + abs(c1 - c2) == 1

    def is_valid_tile(self, ind):
        return ind >= 0 and ind < len(self._map)
    
    def tile_at(self, ind):
        return self._map[ind]

    def army_at(self, ind):
        return self._army[ind]

    def set_tile(self, ind, val):
        self._map[ind] = val
        
    def set_army(self, ind, val):
        self._armies[ind] = val

    def increment_army_at(self, ind):
        self._armies[ind] += 1

    def attack(self, start, end, is50, generals):
        if not self.is_valid_tile(start):
            return False
        
        if not self.is_valid_tile(end):
            return False

        if not self.is_adjacent(start, end):
            return False

        if self.tile_at(end) == TILE_MOUNTAIN:
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
                    self._armies[end] = self._armies[start] - reverse - self._armies[end]
                    self.set_tile(end, self.tile_at[start])

        # Attacking an ally
        else:
            self._armies[end] += self._armies[start] - reserve

            # Take ownership of army only if not a general tile
            if not np.where(generals == end)[0]:
                self.set_tile(end, self.tile_at(start))

        self._armies[start] = reserver

        return True

    def replace_all(self, val1, val2, scale = 1):
        for i in range(len(self._map)):
            if self._map[i] == val1:
                self._map[i] = val2
                self._armies[i] *= scale

    def distance(self, ind1, ind2):
        r1 = ind1 / self.width
        c1 = ind1 % self.width
        r2 = ind2 / self.width
        c2 = ind2 % self.width

        return abs(r1 - r2) + abs(c1 - c2)

                    
                
