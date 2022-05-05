from typing import List
import torch
from torch.distributions.categorical import Categorical
from env import VRPInstance
from .layers import Linear, ProbAttention
import torch.nn as nn


class Decoder1(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prob = ProbAttention(8, input_dim, hidden_dim)
        self.fc = Linear(hidden_dim + 2 + hidden_dim, hidden_dim, bias=False)
        self.fc1 = Linear(hidden_dim, hidden_dim, bias=False)

    def init_evrp_decoder(self, embs, pool, data):
        self.embs = embs
        self.pool = self.fc1(pool)
        self.instances: List[VRPInstance] = [item.instance for item in data]
        self.all_actions = []
        self.action_ids = torch.zeros(len(embs), dtype=torch.long).to(self.device)
        self.dones = [False] * len(embs)
        self.demand_filters = [True] * len(embs)
        self.energy_filters = [True] * len(embs)
        self.current_demands = [instance.capacity for instance in self.instances]
        self.current_energy = [instance.energy_capacity for instance in self.instances]
        self.energy_consumption = [instance.energy_consumption for instance in self.instances]
        self.unsolved_demand_ids = [[id for id, node in enumerate(instance.nodes) if node.is_demand] for instance in self.instances]
        self.none_demand_ids = [[id for id, node in enumerate(instance.nodes) if not node.is_demand] for instance in self.instances]
        
        self.demand_capacity = torch.FloatTensor(self.current_demands)
        self.energy_capacity = torch.FloatTensor(self.current_energy)
        self.masks = self.generate_masks()
    
    def update_state(self, next_action_ids):
        for i, instance in enumerate(self.instances):
            if self.dones[i]:
                continue
            node_id = next_action_ids[i]
            node = instance.nodes[node_id]
            if node.is_demand:
                self.unsolved_demand_ids[i].remove(node_id)
            if self.demand_filters[i]:
                demand_cost = node.demand
                self.current_demands[i] -= demand_cost
                if self.current_demands[i] < 0:
                    self.demand_filters[i] = False
                    self.current_demands[i] = 0
                elif node.is_depot:
                    self.current_demands[i] = instance.capacity
            if self.energy_filters[i]:
                energy_consumption = self.energy_consumption[i]
                prev_node = instance.nodes[self.action_ids[i]]
                energy_cost = prev_node.distance_to(node) * energy_consumption
                self.current_energy[i] -= energy_cost
                if self.current_energy[i] < 0:
                    self.energy_filters[i] = False
                    self.current_energy[i] = 0
                elif not node.is_demand:
                    self.current_energy[i] = instance.energy_capacity
        self.action_ids = next_action_ids
    
    def update_masks(self, node_ids, masks, instance, prev_node, current_demand, current_energy, energy_consumption):
        for id in node_ids:
            node = instance.nodes[id]
            demand_cost = node.demand
            energy_cost = prev_node.distance_to(node) * energy_consumption
            if demand_cost <= current_demand and 0 < energy_cost <= current_energy:
                masks[id] = True
                
    def generate_masks(self):
        all_masks = torch.zeros(self.embs.shape[:-1], dtype=torch.bool)
        for i, instance in enumerate(self.instances):
            masks = all_masks[i]
            if self.dones[i]:
                masks[0] = True
                continue
            current_demand = self.current_demands[i]
            current_energy = self.current_energy[i]
            energy_consumption = self.energy_consumption[i]
            prev_id = self.action_ids[i]
            prev_node = instance.nodes[prev_id]
            unsolved_demand_ids = self.unsolved_demand_ids[i]
            none_demand_ids = self.none_demand_ids[i]
            if self.demand_filters[i] and self.energy_filters[i]:
                self.update_masks(unsolved_demand_ids, masks, instance, prev_node, current_demand, current_energy, energy_consumption)
                if any(masks) or prev_node.is_demand:
                    self.update_masks(none_demand_ids, masks, instance, prev_node, current_demand, current_energy, energy_consumption)
            else:
                self.dones[i] = True
            if not all(masks):
                if len(unsolved_demand_ids) > 0:
                    masks[unsolved_demand_ids] = True
                else:
                    masks[0] = True
            # if prev_node.is_depot and len(unsolved_demand_ids) == 0:
            if prev_node.is_depot and len(self.all_actions) > 1:
                self.dones[i] = True
        return all_masks.to(self.device)
    
    def prepare_input(self):
        inputs = torch.stack([self.embs[i,id,:] for i, id in enumerate(self.action_ids)])
        demands = torch.FloatTensor(self.current_demands) / self.demand_capacity
        energy = torch.FloatTensor(self.current_energy) / self.energy_capacity
        inputs = torch.cat([inputs, demands[:,None].to(self.device), energy[:,None].to(self.device), self.pool], -1)
        inputs = self.fc(inputs)
        return inputs

    def step(self, action_ids=None):
        self.all_actions.append(self.action_ids)
        inputs = self.prepare_input()
        prob = self.prob(inputs, self.embs, ~self.masks)
        pd = Categorical(prob)
        
        if action_ids is None:
            action_ids = pd.sample()
            self.update_state(action_ids)
            self.masks = self.generate_masks()
            log_probs = pd.log_prob(action_ids)
            return log_probs
        else:
            self.update_state(action_ids)
            self.masks = self.generate_masks()
            log_probs = pd.log_prob(action_ids)
            entropy = pd.entropy()
            return log_probs, entropy
        
    def forward(self, embs, pool, data, actions=None):
        self.init_evrp_decoder(embs, pool, data)
        if actions is None:
            log_probs = []
            k = 0
            count = []
            while not all(self.dones):
                log_prob = self.step()
                dones = 1 - torch.FloatTensor(self.dones)
                dones = dones.to(self.device)
                log_prob *= dones
                count.append(dones)
                log_probs.append(log_prob)
                print("k:", k, "   ", end="\r")
                k += 1
            self.all_actions.append(self.action_ids)
            count = torch.stack(count).sum(0)
            actions = torch.stack(self.all_actions).T
            log_probs = torch.sum(torch.stack(log_probs, -1), -1) / count
            return actions, log_probs
        else:
            log_probs = []
            all_entropy = []
            count = []
            for action in actions:
                log_prob, entropy = self.step(action)
                dones = 1 - torch.FloatTensor(self.dones)
                dones = dones.to(self.device)
                log_prob *= dones
                entropy *= dones
                count.append(dones)
                log_probs.append(log_prob)
                all_entropy.append(entropy.mean())
            count = torch.stack(count).sum(0).to(self.device)
            log_probs = torch.sum(torch.stack(log_probs, -1), -1) / count
            all_entropy = torch.sum(torch.stack(all_entropy, -1), -1)
            return all_entropy, log_probs


class Decoder2(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prob = ProbAttention(8, input_dim, hidden_dim)
        self.fc = Linear(hidden_dim, hidden_dim, bias=False)
        self.fc1 = Linear(hidden_dim, hidden_dim, bias=False)

    def init_evrp_decoder(self, embs, pool, data):
        self.embs = embs
        self.pool = self.fc1(pool)
        self.all_solution = [item.solution for item in data]
        self.all_actions = []
        self.action_ids = torch.zeros(len(embs), dtype=torch.long).to(self.device)
        self.dones = [False] * len(embs)
        self.masks = self.generate_masks()
    
    def update_state(self, next_action_ids):
        self.action_ids = next_action_ids
    
    def generate_masks(self):
        all_masks = torch.zeros(self.embs.shape[:-1], dtype=torch.bool)
        for i, solution in enumerate(self.all_solution):
            masks = all_masks[i]
            if self.dones[i]:
                masks[0] = True
                continue
            action_ids = [actions[i].item() for actions in self.all_actions]
            for id in solution:
                if id not in action_ids:
                    masks[id] = True
                else:
                    action_ids.remove(id)
                    masks[id] = False
            if not any(masks):
                masks[0] = True
                self.dones[i] = True
        return all_masks.to(self.device)
    
    def prepare_input(self):
        inputs = torch.stack([self.embs[i,id,:] for i, id in enumerate(self.action_ids)])
        inputs = self.fc(inputs) + self.pool
        return inputs

    def step(self, action_ids=None):
        self.all_actions.append(self.action_ids)
        inputs = self.prepare_input()
        prob = self.prob(inputs, self.embs, ~self.masks)
        pd = Categorical(prob)
        
        if action_ids is None:
            action_ids = pd.sample()
            self.update_state(action_ids)
            self.masks = self.generate_masks()
            log_probs = pd.log_prob(action_ids)
            return log_probs
        else:
            self.update_state(action_ids)
            self.masks = self.generate_masks()
            log_probs = pd.log_prob(action_ids)
            entropy = pd.entropy()
            return log_probs, entropy
        
    def forward(self, embs, pool, data, actions=None):
        self.init_evrp_decoder(embs, pool, data)
        if actions is None:
            log_probs = []
            k = 0
            count = []
            while not all(self.dones):
                log_prob = self.step()
                dones = 1 - torch.FloatTensor(self.dones)
                dones = dones.to(self.device)
                log_prob *= dones
                count.append(dones)
                log_probs.append(log_prob)
                print("k:", k, "    ", end="\r")
                k += 1
            self.all_actions.append(self.action_ids)
            actions = torch.stack(self.all_actions).T
            log_probs = torch.sum(torch.stack(log_probs, -1), -1)
            return actions, log_probs
        else:
            log_probs = []
            all_entropy = []
            count = []
            for action in actions:
                log_prob, entropy = self.step(action)
                dones = 1 - torch.FloatTensor(self.dones)
                dones = dones.to(self.device)
                log_prob *= dones
                entropy *= dones
                count.append(dones)
                log_probs.append(log_prob)
                all_entropy.append(entropy.mean())
            log_probs = torch.sum(torch.stack(log_probs, -1), -1)
            all_entropy = torch.sum(torch.stack(all_entropy, -1), -1)
            return all_entropy, log_probs


class Decoder3(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prob = ProbAttention(8, input_dim, hidden_dim)
        self.fc = Linear(hidden_dim, hidden_dim, bias=False)
        self.fc1 = Linear(hidden_dim, hidden_dim, bias=False)
        self.max_steps = 128

    def init_evrp_decoder(self, embs, pool, masks):
        self.embs = embs
        self.pool = self.fc1(pool)
        self.masks = masks
        self.all_actions = []
        self.action_ids = torch.zeros(len(embs), dtype=torch.long).to(self.device)
        self.dones = [False] * len(embs)
       
    def update_state(self, next_action_ids):
        self.action_ids = next_action_ids
         
    def generate_masks(self):
        for id, action_id in enumerate(self.action_ids):
            self.masks[id][action_id] = False
            if not torch.any(self.masks[id]):
                self.dones[id] = True
                self.masks[id][0] = True
        return self.masks
    
    def prepare_input(self):
        inputs = torch.stack([self.embs[i, id, :] for i, id in enumerate(self.action_ids)])
        inputs = self.fc(inputs) + self.pool
        return inputs

    def step(self, action_ids=None):
        self.all_actions.append(self.action_ids)
        inputs = self.prepare_input()
        prob = self.prob(inputs, self.embs, ~self.masks)
        pd = Categorical(prob)
        if action_ids is None:
            action_ids = pd.sample()
            self.update_state(action_ids)
            self.masks = self.generate_masks()
            log_probs = pd.log_prob(action_ids)
            return log_probs
        else:
            self.update_state(action_ids)
            self.masks = self.generate_masks()
            log_probs = pd.log_prob(action_ids)
            entropy = pd.entropy()
            return log_probs, entropy
        
    def forward(self, embs, pool, masks, actions=None):
        self.init_evrp_decoder(embs, pool, masks)
        if actions is None:
            log_probs = []
            k = 1
            count = []
            for _ in range(self.max_steps):
                log_prob = self.step()
                dones = 1 - torch.FloatTensor(self.dones)
                dones = dones.to(self.device)
                log_prob *= dones
                count.append(dones)
                log_probs.append(log_prob)
                print("\tk:", k, "   ", end="\r")
                k += 1
            self.all_actions.append(self.action_ids)
            count = torch.stack(count).sum(0)
            actions = torch.stack(self.all_actions).T
            log_probs = torch.sum(torch.stack(log_probs, -1), -1) / count
            return actions, log_probs
        else:
            log_probs = []
            all_entropy = []
            count = []
            for action in actions:
                log_prob, entropy = self.step(action)
                dones = 1 - torch.FloatTensor(self.dones)
                dones = dones.to(self.device)
                log_prob *= dones
                entropy *= dones
                count.append(dones)
                log_probs.append(log_prob)
                all_entropy.append(entropy.mean())
            count = torch.stack(count).sum(0).to(self.device)
            log_probs = torch.sum(torch.stack(log_probs, -1), -1) / count
            all_entropy = torch.sum(torch.stack(all_entropy, -1), -1)
            return all_entropy, log_probs


class Decoder(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prob = ProbAttention(8, input_dim, hidden_dim)
        self.fc_node = Linear(hidden_dim, hidden_dim)
        self.fc_pool = Linear(hidden_dim, hidden_dim)
        self.max_steps = 3

    def init_evrp_decoder(self, embs, pool, masks):
        self.embs = embs
        self.pool = self.fc_pool(pool)
        self.masks = masks
        self.all_actions = []
        self.action_ids = torch.zeros(len(embs), dtype=torch.long).to(self.device)
        self.dones = [False] * len(embs)
        self.n_step = 0
       
    def update_state(self, next_action_ids):
        self.action_ids = next_action_ids
         
    def generate_masks(self):
        for id, action_id in enumerate(self.action_ids):
            self.masks[id][action_id] = False
            if not torch.any(self.masks[id]):
                self.dones[id] = True
                self.masks[id][0] = True
        return self.masks
    
    def prepare_input(self):
        inputs = torch.stack([self.embs[i, id, :] for i, id in enumerate(self.action_ids)])
        inputs = self.fc_node(inputs) + self.pool
        return inputs

    def step(self, action_ids=None):
        self.all_actions.append(self.action_ids)
        inputs = self.prepare_input()
        prob = self.prob(inputs, self.embs, ~self.masks)
        pd = Categorical(prob)
        if action_ids is None:
            action_ids = pd.sample()
            self.update_state(action_ids)
            self.masks = self.generate_masks()
            log_probs = pd.log_prob(action_ids)
            return log_probs
        else:
            self.update_state(action_ids)
            self.masks = self.generate_masks()
            log_probs = pd.log_prob(action_ids)
            entropy = pd.entropy()
            return log_probs, entropy
        
    def forward(self, embs, pool, masks, actions=None):
        self.init_evrp_decoder(embs, pool, masks)
        if actions is None:
            log_probs = []
            count = []
            for self.n_step in range(self.max_steps):
                log_prob = self.step()
                dones = 1 - torch.FloatTensor(self.dones)
                dones = dones.to(self.device)
                log_prob *= dones
                count.append(dones)
                log_probs.append(log_prob)
                # print("\tk:", self.n_step+1, "   ", end="\r")
            self.all_actions.append(self.action_ids)
            count = torch.stack(count).sum(0)
            actions = torch.stack(self.all_actions).T
            log_probs = torch.sum(torch.stack(log_probs, -1), -1) / count
            return actions, log_probs
        else:
            log_probs = []
            all_entropy = []
            count = []
            for action in actions:
                log_prob, entropy = self.step(action)
                dones = 1 - torch.FloatTensor(self.dones)
                dones = dones.to(self.device)
                log_prob *= dones
                entropy *= dones
                count.append(dones)
                log_probs.append(log_prob)
                all_entropy.append(entropy.mean())
            count = torch.stack(count).sum(0).to(self.device)
            log_probs = torch.sum(torch.stack(log_probs, -1), -1) / count
            all_entropy = torch.sum(torch.stack(all_entropy, -1), -1)
            return all_entropy, log_probs

