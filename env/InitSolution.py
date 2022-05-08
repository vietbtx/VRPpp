from collections import OrderedDict, deque
from functools import lru_cache
from itertools import combinations
import math
import sys
from typing import List, Tuple
from .VRPNode import Node
from .VRPInstance import VRPInstance
import math
from typing import List
import random

sys.setrecursionlimit(2048)


class Point:

    def __init__(self) -> None:
        self.id: int = None
        self.neighbours_count: int = 0
        self.cluster_ID: int = None
        self.adjacentPoints: List[int] = []

    def copy(self):
        obj = Point()
        obj.id = self.id
        obj.neighbours_count = self.neighbours_count
        obj.cluster_ID = self.cluster_ID
        obj.adjacentPoints = self.adjacentPoints.copy()
        return obj

    def __repr__(self) -> str:
        return f"{self.neighbours_count}"


class InitDBCA:

    def __init__(self, instance: VRPInstance, round_int) -> None:
        self.dbca_OUTLIER = -2
        self.dbca_NOT_CLASSIFIED = -1
        self.counter = 0
        self.lastN = [0, 0, 0, 0]
        self.nextLastNIndex = 0
        self.nV: int = None
        self.dist: List[List[int]] = None
        self.next: List[List[int]] = None
        self.round_int = round_int
        self.read_problem(instance)

    @classmethod
    def from_instance(cls, instance, round_int):
        return cls(instance, round_int)

    def generate_2D_matrix_double(self, n: int, m: int) -> List[List[float]]:
        matrix = []
        for i in range(n):
            arr = []
            for j in range(m):
                arr.append(0.0)
            matrix.append(arr)
        return matrix

    def euclidean_distance(self, i: int, j: int) -> float:
        return self.node_list[i].distance_to(self.node_list[j])
        # xd = self.node_list[i].x - self.node_list[j].x
        # yd = self.node_list[i].y - self.node_list[j].y
        # r = math.sqrt(xd * xd + yd * yd)
        # return r

    def compute_distances(self) -> None:
        for i in range(self.ACTUAL_PROBLEM_SIZE):
            for j in range(self.ACTUAL_PROBLEM_SIZE):
                self.distances[i][j] = self.euclidean_distance(i, j)

    def read_problem(self, instance: VRPInstance):
        self.problem_size = len(instance.depots) + len(instance.demands)
        self.NUM_OF_CUSTOMERS = self.problem_size - 1
        self.MAX_CAPACITY = instance.capacity
        self.BATTERY_CAPACITY = instance.energy_capacity
        self.energy_consumption = instance.energy_consumption
        self.NUM_OF_STATIONS = len(instance.stations)
        self.ACTUAL_PROBLEM_SIZE = self.problem_size + self.NUM_OF_STATIONS
        self.cust_demand = [0] * self.ACTUAL_PROBLEM_SIZE
        self.charging_station = [True] * self.ACTUAL_PROBLEM_SIZE
        self.node_list = instance.nodes
        for id, node in enumerate(instance.nodes):
            self.cust_demand[id] = node.demand
            self.charging_station[id] = node.is_station or node.is_depot
            if node.is_depot:
                self.DEPOT = id
        self.distances = self.generate_2D_matrix_double(self.ACTUAL_PROBLEM_SIZE, self.ACTUAL_PROBLEM_SIZE)
        self.compute_distances()

    def init_evals(self) -> None:
        self.evals = 0

    def init_current_best(self) -> None:
        self.current_best = float('inf')

    def get_energy_per_unit(self) -> float:
        return self.energy_consumption

    @lru_cache
    def get_distance(self, start: int, end: int) -> float:
        self.evals += 1.0 / self.ACTUAL_PROBLEM_SIZE
        return self.distances[start][end]

    def density_connected(self, current_p: int, cluster: int, min_pt: int, points: List[Point]) -> None:
        points[current_p].cluster_ID = cluster
        if points[current_p].neighbours_count < min_pt:
            return
        for next in points[current_p].adjacentPoints:
            if points[next].cluster_ID not in [self.dbca_NOT_CLASSIFIED, self.dbca_OUTLIER]:
                continue
            self.density_connected(next, cluster, min_pt, points)

    def is_charging_station(self, node: int) -> bool:
        return self.charging_station[node]

    def dbca(self) -> List[List[List[int]]]:
        reach = self.BATTERY_CAPACITY / self.get_energy_per_unit()
        epss: List[float] = []
        min_pts: List[int] = []
        epss.append(reach/2)
        epss.append(reach/3)
        for i in range(2, 6):
            epss.append(reach/(i*2))
        epss.append(reach/15)
        epss.append(reach/20)
        for i in range(2, 6):
            min_pts.append(i)
        cluster_sets = []
        for x in range(len(epss)):
            eps = epss[x]
            points_base: List[Point] = []
            for _ in range(self.ACTUAL_PROBLEM_SIZE):
                points_base.append(Point())
            for i in range(self.ACTUAL_PROBLEM_SIZE):
                points_base[i].cluster_ID = self.dbca_NOT_CLASSIFIED
                for j in range(self.ACTUAL_PROBLEM_SIZE):
                    if i == j:
                        continue
                    if self.get_distance(i, j) <= eps:
                        points_base[i].neighbours_count += 1
                        points_base[i].adjacentPoints.append(j)
            for y in range(len(min_pts)):
                min_pt = min_pts[y]
                cluster_idx = -1
                points: List[Point] = []
                cluster_set: List[List[int]] = []
                for i in range(self.ACTUAL_PROBLEM_SIZE):
                    points.append(points_base[i].copy())
                for i in range(self.ACTUAL_PROBLEM_SIZE):
                    if points[i].cluster_ID != self.dbca_NOT_CLASSIFIED:
                        continue
                    if points[i].neighbours_count >= min_pt:
                        cluster_idx += 1
                        self.density_connected(i, cluster_idx, min_pt, points)
                    else:
                        points[i].cluster_ID = self.dbca_OUTLIER
                for i in range(self.ACTUAL_PROBLEM_SIZE):
                    if points[i].cluster_ID != self.dbca_OUTLIER:
                        continue
                    minDist = float('inf')
                    min_node_id = -2
                    for j in range(self.ACTUAL_PROBLEM_SIZE):
                        if i == j:
                            continue
                        if points[j].cluster_ID == self.dbca_OUTLIER:
                            continue
                        dist = self.get_distance(i, j)
                        if dist < minDist:
                            minDist = dist
                            min_node_id = j
                    if min_node_id == -2:
                        cluster_idx += 1
                        points[i].cluster_ID = cluster_idx
                    else:
                        points[i].cluster_ID = points[min_node_id].cluster_ID
                for _ in range(cluster_idx+1):
                    cluster_set.append([])
                for i in range(self.ACTUAL_PROBLEM_SIZE):
                    if points[i].cluster_ID != self.dbca_OUTLIER:
                        cluster_set[points[i].cluster_ID].append(i)
                i = 0
                while i < len(cluster_set):
                    has_customer = False
                    for next in cluster_set[i]:
                        if not self.is_charging_station(next):
                            has_customer = True
                    if not has_customer:
                        del cluster_set[i]
                    i += 1
                i = 0
                while i < len(cluster_set):
                    has_depo = False
                    for next in cluster_set[i]:
                        if next == self.DEPOT:
                            has_depo = True
                    if not has_depo:
                        cluster_set[i].append(self.DEPOT)
                    i += 1
                has_duplicate = False
                for c in cluster_sets:
                    if c == cluster_set:
                        has_duplicate = True
                if has_duplicate:
                    continue
                cluster_sets.append(cluster_set)
        return cluster_sets

    @lru_cache
    def get_customer_demand(self, customer: int) -> int:
        return self.cust_demand[customer]

    def clarke_wright(self, capacitated: bool, clusters: bool = False, node_list: List[int] = []) -> List[int]:
        unusedCustomers: List[int] = []
        if clusters:
            for node in node_list:
                if not self.is_charging_station(node) and node not in unusedCustomers:
                    unusedCustomers.append(node)
        else:
            for i in range(self.ACTUAL_PROBLEM_SIZE):
                if not self.is_charging_station(i) and i not in unusedCustomers:
                    unusedCustomers.append(i)
        subtours: List[List[int]] = []
        while len(unusedCustomers) > 0:
            subtour: List[int] = []
            remaining_capacity = self.MAX_CAPACITY
            maxDist = 0
            for cand in unusedCustomers:
                dist = self.get_distance(0, cand)
                if dist > maxDist:
                    maxDist = int(dist)
                    furthest = cand
            subtour.append(furthest)
            remaining_capacity -= self.get_customer_demand(furthest)
            unusedCustomers.remove(furthest)
            dist_front = maxDist
            dist_back = maxDist
            enough_capacity = True
            while enough_capacity:
                enough_capacity = False
                distImprovement = float('inf')
                front = subtour[0]
                back = subtour[-1]
                at_front = False
                for cand in unusedCustomers:
                    if self.get_customer_demand(cand) <= remaining_capacity or not capacitated:
                        enough_capacity = True
                        dist_candidate_depo = self.get_distance(0, cand)
                        dist_saved_front = self.get_distance(
                            front, cand) - dist_candidate_depo - dist_front
                        dist_saved_back = self.get_distance(
                            back, cand) - dist_candidate_depo - dist_back
                        if dist_saved_front < distImprovement:
                            at_front = True
                            distImprovement = dist_saved_front
                            closest = cand
                            dist_to_depo = dist_candidate_depo
                        if dist_saved_back < distImprovement:
                            at_front = False
                            distImprovement = dist_saved_back
                            closest = cand
                            dist_to_depo = dist_candidate_depo
                if not enough_capacity:
                    break
                if at_front:
                    dist_front = dist_to_depo
                    subtour.insert(0, closest)
                else:
                    dist_back = dist_to_depo
                    subtour.append(closest)
                remaining_capacity -= self.get_customer_demand(closest)
                unusedCustomers.remove(closest)
                if len(unusedCustomers) == 0:
                    enough_capacity = False
                    break
            subtours.append(subtour)
        tmp: List[int] = []
        if capacitated:
            tmp.append(0)
            for tour in subtours:
                for customer in tour:
                    tmp.append(customer)
                tmp.append(0)
        else:
            tmp = subtours[0]
        return tmp

    def getRemainingLoad(self, evrpTour: List[int]) -> int:
        load = 0
        for node in evrpTour:
            if node == 0:
                load = self.MAX_CAPACITY
            else:
                load -= self.get_customer_demand(node)
        return load

    @lru_cache
    def getClosestAFS(self, node: int) -> int:
        minDist = float('inf')
        for i in range(self.ACTUAL_PROBLEM_SIZE):
            if self.is_charging_station(i) and i != node:
                dist = self.get_distance(node, i)
                if dist < minDist:
                    minDist = dist
                    closest = i
        return closest

    def getRemainingBattery(self, evrpTour: List[int]) -> float:
        battery = 0
        for i in range(len(evrpTour)):
            cur = evrpTour[i]
            if i > 0:
                prev = evrpTour[i-1]
                battery -= self.get_energy_consumption(prev, cur)
            if self.is_charging_station(cur):
                battery = self.BATTERY_CAPACITY
        return battery

    @lru_cache
    def get_energy_consumption(self, start: int, end: int) -> float:
        return self.energy_consumption * self.distances[start][end]

    def addAndCheckLastN(self, node: int, reset: int = False) -> bool:
        if reset:
            for i in range(4):
                self.lastN[i] -= 1
            self.counter = 0
        self.counter += 1
        check = True
        if self.counter > 4:
            check = False
            for i in range(2):
                index1 = i % 4
                index2 = (i + 2) % 4
                if self.lastN[index1] != self.lastN[index2]:
                    check = True
                    break
        if check:
            self.lastN[self.nextLastNIndex] = node
            self.nextLastNIndex = (self.nextLastNIndex + 1) % 4
        return check

    @lru_cache
    def getReachableAFSClosestToGoal(self, cur: int, goal: int, battery: float) -> int:
        minDist = float('inf')
        closest = -1
        for i in range(self.ACTUAL_PROBLEM_SIZE):
            if self.is_charging_station(i) and i != cur and battery >= self.get_energy_consumption(cur, i):
                dist = self.get_distance(i, goal)
                if dist < minDist:
                    minDist = dist
                    closest = i
        return closest
    
    def getPath(self, u: int, v: int, afsIds: bool) -> List[int]:
        path: List[int] = []
        if self.next[u][v] == -1:
            return path
        path.append(u)
        while u != v:
            u = self.next[u][v]
            path.append(u)
        if afsIds:
            for i in range(len(path)):
                path[i] = self.AFSs[path[i]]
        return path
    
    def init_matrix(self, default_value: int) -> List[List[int]]:
        matrix = []
        for _ in range(self.nV):
            arr = [default_value] * self.nV
            matrix.append(arr)
        return matrix
    
    def init_floyd_Warshall(self, nV: int) -> None:
        self.nV = nV
        self.dist = self.init_matrix(-1)
        self.next = self.init_matrix(math.inf)
        for i in range(nV):
            for j in range(i, nV):
                if i == j:
                    self.dist[i][j] = 0
                    self.next[i][j] = j
                else:
                    start = self.AFSs[i]
                    goal = self.AFSs[j]
                    consumption = self.get_energy_consumption(start, goal)
                    self.get_distance(start, goal)
                    if consumption <= self.BATTERY_CAPACITY:
                        self.dist[i][j] = consumption
                        self.dist[j][i] = consumption
                        self.next[i][j] = j
                        self.next[j][i] = i

    def planPaths(self) -> None:
        for k in range(self.nV):
            for i in range(self.nV):
                for j in range(self.nV):
                    if self.dist[i][k] + self.dist[k][j] < self.dist[i][j]:
                        self.dist[i][j] = self.dist[i][k] + self.dist[k][j]
                        self.next[i][j] = self.next[i][k]
    
    def initMyStructures(self) -> None:
        self.AFSs = []
        self.CUSTOMERS = []
        self.afsIdMap = OrderedDict()
        afsId = 0
        for i in range(self.ACTUAL_PROBLEM_SIZE):
            if self.is_charging_station(i):
                self.AFSs.append(i)
                self.afsIdMap[i] = afsId
                afsId += 1
            else:
                self.CUSTOMERS.append(i)

        self.init_floyd_Warshall(len(self.AFSs))
        self.planPaths()

    def get_to_depot_possibly_through_afss(self, evrpTour: List[int]) -> bool:
        canReachDepot = self.getRemainingBattery(
            evrpTour) - self.get_energy_consumption(evrpTour[-1], 0) >= 0
        if canReachDepot:
            evrpTour.append(0)
        else:
            closestAFS = self.getClosestAFS(evrpTour[-1])
            afsSubpath = self.getPath(self.afsIdMap[closestAFS], self.afsIdMap[0], True)
            evrpTour.extend(afsSubpath)
        return self.addAndCheckLastN(0)

    def tsp2evrp_zga_relaxed(self, tspTour: List[int]) -> List[int]:
        evrpTour: List[int] = []
        nextId = 0
        evrpTour.append(0)
        while nextId != len(tspTour):
            if self.getRemainingLoad(evrpTour) < self.get_customer_demand(tspTour[nextId]):
                check = self.get_to_depot_possibly_through_afss(evrpTour)
                if not check:
                    break
            else:
                closestAFSToGoal = self.getClosestAFS(tspTour[nextId])
                remainingBattery = self.getRemainingBattery(evrpTour)
                energyToNext = self.get_energy_consumption(evrpTour[-1], tspTour[nextId])
                nextToAFS = self.get_energy_consumption(tspTour[nextId], closestAFSToGoal)
                if remainingBattery - energyToNext >= nextToAFS:
                    evrpTour.append(tspTour[nextId])
                    nextId += 1
                    if not self.addAndCheckLastN(nextId - 1):
                        break
                else:
                    closestAFS = self.getReachableAFSClosestToGoal(evrpTour[-1], tspTour[nextId], self.getRemainingBattery(evrpTour))
                    canReach = self.getRemainingBattery(evrpTour) - self.get_energy_consumption(evrpTour[-1], closestAFS) >= 0
                    if not canReach:
                        continue
                    evrpTour.append(closestAFS)
                    if not self.addAndCheckLastN(closestAFS):
                        break
        self.get_to_depot_possibly_through_afss(evrpTour)
        return evrpTour

    def fitness_evaluation(self, tour: List[int]) -> float:
        tour_length = 0
        for i in range(len(tour) - 1):
            tour_length += self.distances[tour[i]][tour[i+1]]
        if tour_length < self.current_best:
            self.current_best = tour_length
        self.evals += 1
        return tour_length

    def init_from_dbca(self) -> List[int]:
        cluster_sets = self.dbca()
        evrp_tours: List[List[int]] = []
        for x in range(len(cluster_sets)):
            cluster_set = cluster_sets[x]
            evrp_tour: List[int] = []
            for y in range(len(cluster_set)):
                cluster = cluster_set[y]
                initTour = self.clarke_wright(True, True, cluster)
                tmp_evrp_tour = self.tsp2evrp_zga_relaxed(initTour)
                evrp_tour.extend(tmp_evrp_tour)
            last = -1
            i = 0
            while i < len(evrp_tour):
                if last == 0 and evrp_tour[i] == 0:
                    del evrp_tour[i]
                    i -= 1
                last = evrp_tour[i]
                i += 1
            evrp_tours.append(evrp_tour)
        minEval = float('inf')
        bestTour: List[int] = []
        for tour in evrp_tours:
            eval = self.fitness_evaluation(tour)
            if eval < minEval:
                minEval = eval
                bestTour = tour
        return bestTour, minEval

    def init_solution(self):
        self.init_evals()
        self.init_current_best()
        self.initMyStructures()
        current_solution, score = self.init_from_dbca()
        return current_solution, score


class Tour:

    def __init__(self, nodes):
        self.nodes = nodes

    def append(self, node):
        self.nodes.append(node)

    @classmethod
    def add(cls, station_nodes):
        cls.station_nodes = station_nodes
        return cls

    @staticmethod
    @lru_cache
    def angle_to(node_a, node_b) -> float:
        return math.atan2(node_b.y - node_a.y, node_b.x - node_a.x)

    @staticmethod
    def distance_to(node_a, node_b) -> float:
        return node_a.distance_to(node_b)

    @staticmethod
    @lru_cache
    def nearest_station_to(node_a, node_b):
        best_cost = math.inf
        for station in Tour.station_nodes:
            cost = Tour.distance_to(node_a, station) + \
                Tour.distance_to(station, node_b)
            if cost < best_cost:
                best_cost = cost
                best_station = station
        extra_cost = best_cost - Tour.distance_to(node_a, node_b)
        return best_station, extra_cost

    def is_energy_feasible(self, battery_capacity, energy_consumption):
        current_energy = battery_capacity
        prev_node = self.nodes[0]
        for node in self.nodes[1:] + self.nodes[:1]:
            current_energy -= energy_consumption * \
                Tour.distance_to(prev_node, node)
            if current_energy < 0:
                return False
            if node.is_station:
                current_energy = battery_capacity
            prev_node = node
        return True

    def is_demand_feasible(self, max_capacity: int) -> bool:
        total_demand = sum(node.demand for node in self.nodes)
        return total_demand <= max_capacity

    def is_valid(self, ctx):
        return self.is_energy_feasible(ctx) and self.is_demand_feasible(ctx)

    def __repr__(self):
        return f"{self.nodes}"

    def compute_distance(self) -> float:
        distance = 0
        prev_node = self.nodes[0]
        for node in self.nodes[1:] + self.nodes[:1]:
            distance += Tour.distance_to(prev_node, node)
            prev_node = node
        return distance

    def clone(self):
        return Tour(self.nodes.copy())


class InitClockHand:

    def __init__(self, instance: VRPInstance, round_int=True) -> None:
        self.nodes = instance.nodes
        self.demand_nodes = [node for node in self.nodes if node.demand > 0]
        self.depot = [node for node in self.nodes if node.is_depot][0]
        self.station_nodes = [node for node in self.nodes if node.is_station]
        self.TWOPI = 2*math.pi
        self.max_capacity = instance.capacity
        self.battery_capacity = instance.energy_capacity
        self.energy_consumption = instance.energy_consumption
        self.tool = Tour.add(self.station_nodes)
        self.round_int = round_int

    @classmethod
    def from_instance(cls, instance, round_int):
        return cls(instance, round_int)

    def repair_demand(self, nodes):
        result = []
        queue = deque()
        queue.append(nodes)
        while len(queue) > 0:
            check_nodes = queue.pop()
            if Tour(check_nodes).is_demand_feasible(self.max_capacity):
                result.append(check_nodes)
            else:
                node1, node2 = self.bifurcate(check_nodes)
                queue.append(node1)
                queue.append(node2)
        return result
    
    def clip2pi(self, theta):
        two_pi = 2*math.pi
        if theta < 0:
            return theta + math.ceil(-theta/two_pi) * two_pi
        return theta % two_pi
    
    def bifurcate(self, nodes):
        nodes = [node for node in nodes if not node.is_station]
        theta_nodes = [(node, self.depot.angle_to(node)) for node in nodes]
        theta_nodes.sort(key=lambda x: x[1])
        max_gap_value = None
        for i in range(len(nodes)):
            prev_i = (i + len(nodes) - 1) % len(nodes)
            gap = self.clip2pi(theta_nodes[i][1] - theta_nodes[prev_i][1])
            if max_gap_value is None or gap > max_gap_value:
                max_gap_value = gap
                offset_theta = theta_nodes[i][1]

        theta_nodes.sort(key=lambda x: self.clip2pi(x[1] - offset_theta))
        theta_nodes = [node for node, _ in theta_nodes]
        midpoint = random.randrange(len(theta_nodes))
        node1 = [self.depot] + theta_nodes[:midpoint]
        node2 = [self.depot] + theta_nodes[midpoint:]
        return node1, node2

    def partition(self, nodes):
        best_cost = None
        for node in nodes:
            theta = self.tool.angle_to(self.depot, node)
            candidate = self.clock_hand_partition(theta, self.demand_nodes)
            tours: List[Tour] = []
            for tour in candidate:
                result = self.resolve_tour(tour)
                if result is None:
                    tours = None
                    break
                tours += result
            if tours is None:
                continue
            cost = sum(self.compute_distance(tuple(tour.nodes)) for tour in tours)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_candidate = candidate
        return best_candidate

    def clock_hand_partition(self, theta: float, nodes) -> List[Tour]:
        if theta is not None:
            def theta_func(node):
                return (self.tool.angle_to(self.depot, node) + self.TWOPI - theta) % self.TWOPI
            theta_nodes = sorted(nodes, key=theta_func)
        else:
            theta_nodes = nodes
        result: List[Tour] = []
        tour = Tour([self.depot])
        result.append(tour)
        current_demand = 0
        for node in theta_nodes:
            if current_demand + node.demand > self.max_capacity:
                tour = Tour([self.depot])
                current_demand = 0
                result.append(tour)
            current_demand += node.demand
            tour.append(node)
        return result

    @lru_cache
    def solve_base_tsp(self, nodes) -> Tour:
        min_x = min(node.x for node in nodes)
        min_y = min(node.y for node in nodes)
        max_x = max(node.x for node in nodes)
        max_y = max(node.y for node in nodes)
        centroid = Node(-1, (min_x+max_x)/2, (min_y+max_y)/2)
        theta_nodes = sorted(
            nodes, key=lambda x: self.tool.angle_to(centroid, x))
        depot_id = theta_nodes.index(self.depot)
        result = theta_nodes[depot_id:] + theta_nodes[:depot_id]
        return result

    @lru_cache
    def mod(self, i: int, n: int) -> int:
        return (i + n) % n

    def compute_distance(self, nodes):
        distance = 0
        prev_node = nodes[0]
        for node in nodes[1:] + nodes[:1]:
            distance += self.tool.distance_to(prev_node, node)
            prev_node = node
        return distance

    def solve_tsp(self, sets, max_gap: int = 4) -> List[Tour]:
        result = []
        for nodes in sets:
            tour_nodes = self.solve_base_tsp(tuple(nodes))
            tour = Tour(tour_nodes)
            n = len(tour.nodes)
            if n <= 3:
                result.append(tour)
                continue
            tsp_nodes = tour.nodes
            tsp_cost = self.compute_distance(tuple(tour.nodes))
            while True:
                any_gain = False
                for i in range(n):
                    i0 = i % n
                    n0 = tsp_nodes[i0]
                    actual_max = min(max_gap, len(tsp_nodes)//2)
                    for gap in range(2, actual_max+1):
                        i1 = (i + 1) % n
                        i2 = (i + gap) % n
                        i3 = (i + gap + 1) % n
                        n1 = tsp_nodes[i1]
                        n2 = tsp_nodes[i2]
                        n3 = tsp_nodes[i3]
                        delta = self.tool.distance_to(n0, n2) \
                            + self.tool.distance_to(n1, n3) \
                            - self.tool.distance_to(n0, n1) \
                            - self.tool.distance_to(n2, n3)
                        if delta < -0.000001:
                            segment = []
                            x = i1
                            while x != self.mod(i2+1, n):
                                segment.append(tsp_nodes[x])
                                x = self.mod(x+1, n)
                            si = len(segment) - 1
                            x = i1
                            while x != self.mod(i2+1, n):
                                tsp_nodes[x] = segment[si]
                                x = self.mod(x+1, n)
                                si -= 1
                            tsp_cost += delta
                            any_gain = True
                if not any_gain:
                    break
            depot_id = tsp_nodes.index(self.depot)
            tour_nodes = tsp_nodes[depot_id:] + tsp_nodes[:depot_id]
            result.append(Tour(tour_nodes))
        return result

    def resolve_tour(self, tour: Tour):
        nodes = [node for node in tour.nodes if not node.is_station or node.is_depot]
        demand_feasible_sets = self.repair_demand(nodes)
        demand_feasible_tours = self.solve_tsp(demand_feasible_sets)
        energy_feasible_tours = self.repair_energy(demand_feasible_tours)
        return energy_feasible_tours

    @lru_cache
    def _repair_energy(self, tour_nodes):
        tour = Tour(tour_nodes)
        nodes = [node for node in tour.nodes if not node.is_station]
        best_extra_cost = None
        best_repaired_tour_nodes = None
        for station_count in range(1, len(tour.nodes) + 1):
            for insertion_points in combinations(range(len(tour.nodes)), station_count):
                total_extra_cost = 0
                repaired_tour_node = []
                next_insertion_id = 0
                for node_id in range(len(nodes)):
                    repaired_tour_node.append(tour.nodes[node_id])
                    if next_insertion_id < len(insertion_points) and node_id == insertion_points[next_insertion_id]:
                        node1 = tour.nodes[node_id]
                        node2 = tour.nodes[(node_id + 1) % len(nodes)]
                        best_stations, extra_cost = self.tool.nearest_station_to(
                            node1, node2)
                        repaired_tour_node.append(best_stations)
                        total_extra_cost += extra_cost
                        next_insertion_id += 1
                if not Tour(repaired_tour_node).is_energy_feasible(self.battery_capacity, self.energy_consumption):
                    continue
                if best_extra_cost is None or total_extra_cost < best_extra_cost:
                    best_repaired_tour_nodes = repaired_tour_node
                    best_extra_cost = total_extra_cost
            if best_repaired_tour_nodes is not None and (station_count == 0 or station_count >= 2):
                break
        return best_repaired_tour_nodes

    def repair_energy(self, tours: Tuple[Tour]) -> List[Tour]:
        result = []
        for tour in tours:
            if tour.is_energy_feasible(self.battery_capacity, self.energy_consumption):
                if any(not node.is_station for node in tour.nodes):
                    result.append(tour)
                continue
            best_repaired_tour_nodes = self._repair_energy(tuple(tour.nodes))
            if best_repaired_tour_nodes is None:
                return
            best_tour = Tour(best_repaired_tour_nodes)
            if any(not node.is_station for node in best_tour.nodes):
                result.append(best_tour)
        return result

    def _init_solution(self, nodes):
        partitions = self.partition(nodes)
        tours = []
        for tour in partitions:
            tours += self.resolve_tour(tour)
        return tours

    def init_solution(self) -> Tuple[List[int], float]:
        solution = self._init_solution(self.demand_nodes)
        score = sum(tour.compute_distance() for tour in solution)
        solution = [node.id for tour in solution for node in tour.nodes] 
        solution = [id for i, id in enumerate(solution) if solution[i] != solution[i-1]] + [self.depot.id]
        return solution, score
