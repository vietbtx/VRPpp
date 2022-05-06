import torch
import numpy as np
from typing import Dict, List
from .Score import Score
from .utils import set_seed
from .VRPInstance import VRPInstance
from torch_geometric.data import Data


class VRP:

    def __init__(self, instances: List[VRPInstance], args):
        self.instances = instances
        self.args = args
        self.instance: VRPInstance = None
        self.instance_lens = np.array([len(instance.nodes) for instance in self.instances])
        self.instance_names = [instance.name for instance in self.instances]
        self.instance_count = self.init_instance_count()
        self.best_scores: Dict[str, Score] = {}
        set_seed(args.seed)
        self.reset()

    def reset(self, sub_instance=None):
        score = None
        if self.instance is None or sub_instance is None or sub_instance.vrp is None:
            if self.instance is not None:
                score = self.instance.evaluation(self.instance.solution)
                score = Score(self.instance.name, score, self.instance.solution)
                best_score = self.best_scores.get(self.instance.name, None)
                if best_score is not None and best_score.score < score.score:
                    self.instance.init_solution = best_score.solution
                else:
                    self.instance.init_solution = self.instance.solution
                self.instance.solution = []
            self.instance: VRPInstance = np.random.choice(self.instances, p=self.probs)
            self.instance.create_sub_instance(self.n_extend_tours)
        self.sub_solution = self.instance.sub_instance.init_solution
        self.offspring = self.sub_solution
        self.sub_score = self.instance.sub_instance.evaluation(self.sub_solution)
        self.count = 0
        self.score_scale = self.sub_score / self.args.reward_norm
        return score
    
    def init_instance_count(self):
        return np.array([self.args.max_count for _ in self.instances])
    
    @property
    def probs(self):
        probs = self.instance_lens * self.instance_count
        probs = probs/sum(probs)
        return probs

    @property
    def n_extend_tours(self):
        n = self.instance.vehicles
        n = int(n * self.args.extend_tour_rate)
        n = min(n, self.args.min_extend_tours)
        n = max(n, self.args.max_extend_tours)
        return n

    def get_instance(self, name):
        for instance in self.instances:
            if instance.name == name:
                return instance

    def is_done(self):
        return self.count >= self.args.max_steps

    def setup_instance_count(self, instance_count):
        self.instance_count = self.init_instance_count()
        for name, count in instance_count.items():
            id = self.instance_names.index(name)
            self.instance_count[id] = max(self.instance_count[id]-count, 0)

    def step(self, action, arr, worker_id):
        action, instance_count = action
        self.setup_instance_count(instance_count)
        solution = self.sub_solution.copy()
        action = action[1:]
        i = 0
        while i+2 < len(action):
            s, e = action[i:i+2]
            if s > e: s, e = e, s
            solution = solution[:s] + list(reversed(solution[s:e+1])) + solution[e+1:]
            i += 1
        if solution[0] != 0:
            solution = [0] + solution
        while solution[-1] == 0:
            solution = solution[:-1]
        solution.append(0)

        self.offspring, solution, score = self.instance.sub_instance.step(solution, arr, worker_id)

        reward = (self.sub_score - score) / self.score_scale
        if self.sub_score > score:
            self.sub_score = score
            self.sub_solution = solution
        
        self.count += 1

        if (self.count + 1) % self.args.n_steps == 0 and self.args.imitation_rate > 0:
            self.offspring = self.sub_solution  # for imitation
        
        done = self.is_done()
        instance_score = None
        if done:
            self.instance.done(self.sub_solution)
            self.instance.create_sub_instance(self.n_extend_tours)
            instance_score = self.reset(self.instance.sub_instance)
        return reward, done, instance_score

    def state(self):
        sub_instance = self.instance.sub_instance
        sub_nodes = sub_instance.nodes

        edge_index = []
        edge_features = []
        sol_ids = self.offspring
        prev_id = sol_ids[0]
        for id in sol_ids[1:]:
            edge_index.append([prev_id, id])
            edge_index.append([id, prev_id])
            prev_node = sub_nodes[prev_id]
            node = sub_nodes[id]
            energy_cost = prev_node.distance_to(node) * sub_instance.energy_consumption
            energy_cost = energy_cost / sub_instance.energy_capacity
            edge_features.append([energy_cost, prev_node.angle_to(node)])
            edge_features.append([energy_cost, node.angle_to(prev_node)])
            prev_id = id

        edge_index = torch.LongTensor(edge_index).T
        edge_features = torch.FloatTensor(edge_features)

        node_pos = []
        node_features = []
        for node in sub_nodes:
            node_pos.append([node.x, node.y])
            node_features.append([node.demand])
        node_pos = torch.FloatTensor(node_pos)
        node_features = torch.FloatTensor(node_features)

        node_pos = node_pos/node_pos.max()
        node_features = node_features/sub_instance.capacity

        mask_ids = [id for id in self.sub_solution if self.instance.sub_instance.nodes[id].is_demand]
        
        data = Data(
            x=node_features,
            pos=node_pos,
            edge_index=edge_index,
            edge_attr=edge_features,
            solution=self.sub_solution,
            mask_ids=mask_ids
        )
        return data

    def update_solution(self, scores):
        self.best_scores = scores
