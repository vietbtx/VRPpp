from .utils import angle_between, distance_between


class Node:

    def __init__(self, id, x, y, demand=0, is_depot=False, edge_weight=None, round_int=False):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.is_depot = is_depot
        self.edge_weight = edge_weight[id-1] if edge_weight is not None and 0 < id <= len(edge_weight) else None
        self.round_int = round_int
    
    def __repr__(self) -> str:
        return f"{self.id}"

    def set_distance(self, other, distance):
        if self.id >= other.id:
            self.distance_map[other.id] = distance
        else:
            other.distance_map[self.id] = distance

    def distance_to(self, other):
        if self.id == other.id:
            return 0
        elif self.id > other.id:
            if self.edge_weight is not None:
                distance = self.edge_weight[other.id]
            else:
                distance = distance_between(self.x, self.y, other.x, other.y)
            if self.round_int:
                distance = int(distance + 0.5)
            return distance
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
    