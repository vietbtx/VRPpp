
from functools import lru_cache
import os
import numpy as np
from PIL import Image
from .VRPNode import Node
from typing import Dict, List
from .utils import convert_solution_to_tours, generate_init_tours, plot_solution, read_instance
from cvrp_cpp import CVRP
from evrp_cpp import EVRP


def read_nodes(node_coord_section, demand_section, round_int=False):
    nodes: Dict[int, Node] = {}
    for id, x, y in node_coord_section:
        nodes[id-1] = Node(id-1, x, y, round_int=round_int)
    for id, demand in demand_section:
        nodes[id-1].demand = demand
    nodes[0].is_depot = True
    return nodes.values()


class VRPInstance:

    def __init__(self, mode, capacity, energy_capacity, energy_consumption, nodes: List[Node], name, args):
        self.mode = mode
        self.name = name
        self.capacity = capacity
        self.energy_capacity = energy_capacity
        self.energy_consumption = energy_consumption
        self.depots = [node for node in nodes if node.is_depot]
        self.demands = [node for node in nodes if node.is_demand]
        self.stations = [node for node in nodes if node.is_station]
        self.args = args
        self.init_solution = None
        self.vrp = None
        self.vehicles = 0
        self.solution = []
        self.setup()
    
    def setup(self):
        if len(self.demands) > 0 and not self.name.startswith("clone_"):
            if not self.name.startswith("sub_"):
                name = os.path.normpath(os.path.join(self.args.data_folder, self.name))
                init_mode = 'clockhand'
                if self.mode == "EVRP" and self.args.algo == "VNS":
                    init_mode = 'default'
                self.init_solution = generate_init_tours(self, name, init_mode, self.args.round_int)
                self.vehicles = self.init_solution.count(0) - 1
            else:
                self.vrp = self.make_env()
                if self.mode == "EVRP" and self.args.algo != "VNS":
                    self.vrp_repairer = self.make_env("VNS")
                else:
                    self.vrp_repairer = None

    def make_env(self, algo=None):
        if algo is None:
            algo = self.args.algo
        n_dimension = len(self.depots) + len(self.demands)
        n_stations = len(self.stations)
        if algo == "HGS":
            vrp = CVRP(self.capacity, self.depots + self.demands, self.args.seed, self.vehicles, self.args.round_int)
        else:
            if n_stations == 0:
                self.stations.append(Node(-1, 0, 0))
            vrp = EVRP(n_dimension, n_stations, self.capacity, self.energy_capacity, self.energy_consumption, self.nodes, self.args.seed, self.args.round_int)
        return vrp
                    
    def clone(self):
        return VRPInstance(
            self.capacity, 
            self.energy_capacity,
            self.energy_consumption,
            self.nodes,
            f"clone_{self.name}",
            self.args
        )
    
    def create_sub_instance(self, n_extend_tours=15):
        unsolved_demands = [node for id, node in enumerate(self.nodes) if id not in self.solution and node.is_demand]
        if len(unsolved_demands) == 0:
            self.sub_instance: VRPInstance = None
            return
        init_tours = convert_solution_to_tours(self.nodes, self.init_solution)
        init_tours = [tour for tour in init_tours if any(node in unsolved_demands for node in tour)]
        n = len(init_tours)
        id = 0
        sub_ids = [id]
        for i in range(1, n_extend_tours+1):
            sub_ids.append((n+id-i) % n)
            sub_ids.append((id+i) % n)
        sub_ids = set(sub_ids)
        demands = [node for id in sub_ids for node in init_tours[id] if node in unsolved_demands]
        self.sub_instance = VRPInstance(
            self.mode,
            self.capacity, 
            self.energy_capacity, 
            self.energy_consumption,
            self.depots + demands + self.stations,
            f"sub_{self.name}",
            self.args
        )
        if self.args.algo == "HGS":
            self.sub_instance.init_solution = self.sub_instance.vrp.init_solution()
            self.sub_instance.init_solution = self.sub_instance.repair_solution(self.sub_instance.init_solution)
        else:
            init_solution = [self.sub_instance.nodes.index(node) for id in sub_ids for node in init_tours[id][1:] if not node.is_demand or node in unsolved_demands]
            init_solution = [init_solution[-1]] + init_solution
            _, self.sub_instance.init_solution, _ = self.sub_instance.step(init_solution)
        self.sub_instance.vehicles = self.sub_instance.init_solution.count(0) - 1
    
    def done(self, solution):
        nodes = [self.sub_instance.nodes[id] for id in solution]
        solution = [self.nodes.index(node) for node in nodes]
        tours = convert_solution_to_tours(self.nodes, solution)
        if len(tours) > 4:
            tours = tours[2:-2]
        solved_tour = [node for tour in tours for node in tour[1:]]
        solved_tour = [self.nodes.index(node) for node in solved_tour]
        if len(self.solution) == 0:
            self.solution.append(solved_tour[-1])
        self.solution += solved_tour
    
    def step(self, solution, arr=None, worker_id=None):
        offspring = self.vrp.step(solution)
        if arr is not None and worker_id is not None and self.args.imitation_rate > 0:
            arr[worker_id] = 1
            while any(x == 0 for x in arr):
                self.vrp.sub_step()
            offspring = self.vrp.get_offspring()
        solution = self.vrp.get_best_solution()
        solution = self.repair_solution(solution)
        score = self.evaluation(solution, self.init_solution)
        return offspring, solution, score
    
    def repair_solution(self, solution):
        if self.vrp_repairer is not None:
            self.vrp_repairer.step(solution)
            solution = self.vrp_repairer.get_best_solution()
        return solution
    
    def _step(self, solution, module):
        repaired_solution = module.step(solution)
        try:
            score = self.evaluation(repaired_solution)
        except:
            score = None
        return repaired_solution, score

    def load_image(self, filename) :
        img = Image.open(filename)
        img.load()
        data = np.asarray(img, dtype="int32")
        return data

    def plot(self):
        score = self.evaluation(self.solution)
        tours = convert_solution_to_tours(self.nodes, self.solution)
        fig = plot_solution(self.nodes, tours, self.name, score)
        return fig
    
    def save_plot(self, fig, folder="graphs"):
        os.makedirs(folder, exist_ok=True)
        fig.write_image(f'{folder}/{self.name}.pdf', width=640, height=640)
        
    @classmethod
    def from_path(cls, path, args, name=None):
        data = read_instance(path)
        mode = data.get("TYPE")
        if mode != "EVRP" and args.algo == "VNS":
            id, _, _ = data["NODE_COORD_SECTION"][-1]
            data["NODE_COORD_SECTION"].append([id+1, 0, 0])
        nodes = read_nodes(data["NODE_COORD_SECTION"], data["DEMAND_SECTION"], args.round_int)
        if name is None:
            name = os.path.split(path)[-1]
        capacity = data["CAPACITY"]
        energy_capacity = data.get("ENERGY_CAPACITY", int(1e8))
        energy_consumption = data.get("ENERGY_CONSUMPTION", 1.0)
        return cls(mode, capacity, energy_capacity, energy_consumption, nodes, name, args)
    
    def evaluation(self, solution, default_solution=None):
        try:
            return self._evaluation(tuple(solution))
        except:
            if default_solution is None:
                raise
            return self._evaluation(tuple(default_solution))

    @lru_cache
    def _evaluation(self, solution):
        nodes: List[Node] = self.nodes
        total_distance = 0
        demand_capacity = self.capacity
        energy_capacity = self.energy_capacity
        energy_consumption = self.energy_consumption
        is_valid = True
        demand_nodes = []
        prev_node_id = solution[0]
        for node_id in solution[1:]:
            p_node = nodes[prev_node_id]
            node = nodes[node_id]
            distance = p_node.distance_to(node)
            total_distance += distance
            energy_capacity -= distance * energy_consumption
            demand_capacity -= node.demand
            if prev_node_id == node_id or energy_capacity < 0 or demand_capacity < 0:
                is_valid = False
                break
            if not node.is_demand:
                if node.is_depot:
                    demand_capacity = self.capacity
                energy_capacity = self.energy_capacity
            else:
                demand_nodes.append(node)
            prev_node_id = node_id
        if is_valid:
            is_valid = set(demand_nodes) == set(self.demands) and len(demand_nodes) == len(self.demands)
        assert is_valid
        return total_distance
    
    @property
    def nodes(self) -> List[Node]:
        return self.depots + self.demands + self.stations
    
    def save(self, folder="logs"):
        os.makedirs(folder, exist_ok=True)
        score = self.evaluation(self.solution)
        with open(f"{folder}/score_{self.name}", "a") as f:
            f.write(f"score: {score} - {self.solution}\n")
    
    def save_instance(self, file_name, vehicles=None):
        n_dimension = len(self.depots) + len(self.demands)
        n_stations = len(self.stations)
        with open(file_name, "w") as f:
            f.write(f"Name : \tRandom instance\n")
            f.write(f"COMMENT : \tModified by VietBT.\n")
            f.write(f"TYPE : \t{self.mode}\n")
            # f.write(f"OPTIMAL_VALUE : \t0\n")
            if vehicles:
                f.write(f"VEHICLES : \t{vehicles}\n")
            f.write(f"DIMENSION : \t{n_dimension}\n")
            if len(self.stations) > 1:
                f.write(f"STATIONS : \t{n_stations}\n")
            f.write(f"CAPACITY : \t{self.capacity}\n")
            if len(self.stations) > 1:
                f.write(f"ENERGY_CAPACITY : \t{self.energy_capacity}\n")
                f.write(f"ENERGY_CONSUMPTION : \t{self.energy_consumption}\n")
            f.write(f"EDGE_WEIGHT_TYPE : \tEUC_2D\n")
            f.write(f"NODE_COORD_SECTION\n")
            for id, node in enumerate(self.nodes):
                f.write(f"{id+1} {node.x} {node.y}\n")
            f.write(f"DEMAND_SECTION\n")
            for id, node in enumerate(self.depots + self.demands):
                f.write(f"{id+1} {node.demand}\n")
            if len(self.stations) > 1:
                f.write(f"STATIONS_COORD_SECTION\n")
                for node in self.stations:
                    f.write(f"{self.nodes.index(node)+1}\n")
            f.write(f"DEPOT_SECTION\n")
            f.write(f"1\n")
            f.write(f"-1\n")
            f.write(f"EOF\n")

    def __repr__(self) -> str:
        s = f"{self.name}"
        if not self.name.startswith("clone_"):
            s += f"\n\t- CAPACITY: {self.capacity}"
            s += f"\n\t- ENERGY_CAPACITY: {self.energy_capacity}"
            s += f"\n\t- ENERGY_CONSUMPTION: {self.energy_consumption}"
            s += f"\nNODES: {self.nodes}"
        return s
    