from .utils import angle_between, distance_between


class Node:

    def __init__(self, id, x, y, demand=0, is_depot=False, round_int=False):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.is_depot = is_depot
        self.round_int = round_int
    
    def __repr__(self) -> str:
        return f"{self.id}"
        # return f"{self.id}({self.x},{self.y})"
    
    def distance_to(self, other):
        if self.id >= other.id:
            d = distance_between(self.x, self.y, other.x, other.y)
            if self.round_int:
                d = int(d + 0.5)
            return d
        else:
            return other.distance_to(self)
    
    def angle_to(self, other):
        if self.id >= other.id:
            return angle_between(self.x, self.y, other.x, other.y)
        else:
            return other.angle_to(self)
    
    @property
    def is_station(self):
        return not self.is_depot and self.demand == 0
    
    @property
    def is_demand(self):
        return self.demand > 0
    