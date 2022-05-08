import os
import numpy as np
import torch
from env import VectorizedVRP
from env.utils import load_scores, save_scores, swap_and_flatten
from network.model import Policy
from torch.utils.tensorboard import SummaryWriter


class Runner:

    def __init__(self, env:VectorizedVRP, policy:Policy, log_name: str, args):
        self.env = env
        self.policy = policy
        self.args = args
        self.game = env.env
        self.n_envs = env.n_envs
        self.n_steps = args.n_steps
        self.n_instances = len(self.game.instances)
        self.gamma = args.gamma
        self.lam = args.lam
        self.reward_norm = 1.0
        self.max_count = args.max_count
        self.prev_obs = []
        self.prev_rewards = []
        self.prev_actions = []
        self.prev_values = []
        self.prev_dones = []
        self.prev_log_probs = []
        self.dones = [False]*self.n_envs
        self.games_done = 0
        self.log_name = log_name
        self.log_path = "logs"
        self.log_graph_path = "graphs"
        os.makedirs(f'{self.log_path}', exist_ok=True)
        os.makedirs(f'{self.log_graph_path}', exist_ok=True)
        self.best_scores = load_scores(f"{self.log_path}/{log_name}")
        self.writter = SummaryWriter(f"{self.log_path}/{log_name}")
        self.step = 0
        self.instance_count = {}
        self.history = []
        self.is_new_best_score = False
        self.process_rate = 0
    
    def print_best_score(self):
        names = []
        for name in reversed(self.history):
            if name not in names:
                names.append(name)
                if len(names) > 20:
                    break
        try:
            names.sort(key=lambda x: int(x.split("-")[1][1:]))
        except:
            names.sort()
        print("-"*20)
        print(f"Processed ... {self.process_rate*100:.2f}%")
        for name in names:
            print(f"   - {self.best_scores[name]}\t{self.instance_count.get(name, 0)}")
        print("-"*20)

    def process_score(self, score):
        if score is None:
            return
        score_step = self.instance_count.get(score.name, 0)
        self.writter.add_scalar(f"scores/{score.name}", score.score, score_step)
        self.instance_count[score.name] = score_step + 1
        if score.name not in self.best_scores or self.best_scores[score.name].score > score.score:
            self.is_new_best_score = True
            self.best_scores[score.name] = score
            if self.args.imitation_rate > 0:
                self.env.update_solution(self.best_scores)
            self.history.append(score.name)
            print("New best score:", score)
            # instance = self.game.get_instance(score.name)
            # instance.solution = score.solution
            # instance.save(f"{self.log_path}/{self.log_name}", score.score)
            save_scores(self.best_scores, f"{self.log_path}/{self.log_name}")
            # fig = instance.plot()
            # instance.save_plot(fig, f"{self.log_graph_path}/{self.log_name}")
            # fig.data = []
            # fig.layout = {}
            self.writter.add_scalar(f"best_scores/{score.name}", score.score, score_step)
        
        processed = sum([min(count, self.max_count) for count in self.instance_count.values()])
        total = self.max_count * self.n_instances
        self.process_rate = processed/total
        self.writter.add_scalar(f"process", self.process_rate, self.step)

    def run(self):
        mb_obs, mb_actions, mb_values, mb_log_probs, mb_rewards, mb_dones = [], [], [], [], [], []
        for i in range(len(self.prev_obs)):
            mb_obs.append(self.prev_obs[i])
            mb_actions.append(self.prev_actions[i])
            mb_values.append(self.prev_values[i])
            mb_log_probs.append(self.prev_log_probs[i])
            mb_rewards.append(self.prev_rewards[i])
            mb_dones.append(self.prev_dones[i])
        if len(self.prev_obs) == 1:
            end_len = self.n_steps
        else:
            end_len = self.n_steps-1
        for step in range(self.n_steps):
            print(f"Step: {step+1}")
            obs = self.env.get_current_states()
            with torch.no_grad():
                actions, values, log_probs = self.policy.forward(obs)
                actions = actions.cpu().detach().tolist()
                values = values.cpu().detach().numpy()
                log_probs = log_probs.cpu().detach().numpy()
            
            _actions = [(action, self.instance_count) for action in actions]
            rewards, dones, info = self.env.step(_actions)
            self.step += 1
            self.writter.add_scalar(f"steps", self.step, self.step)

            for score in info:
                try:
                    self.process_score(score)
                except:
                    pass
            
            self.dones = dones
            mb_obs.append(obs)
            mb_actions.append(actions)
            mb_values.append(values)
            mb_log_probs.append(log_probs)
            mb_dones.append(list(dones))
            mb_rewards.append(np.zeros((self.n_envs,)))
            for i in range(self.n_envs):
                reward = rewards[i]
                mb_rewards[-1][i] = reward / self.reward_norm
                if dones[i] == True:
                    self.games_done += 1
        
        if self.is_new_best_score:
            self.print_best_score()
            torch.save(self.policy.state_dict(), f"{self.log_path}/{self.log_name}/model.pt")
            self.is_new_best_score = False

        if self.process_rate >= 1:
            return

        if not self.policy.training:
            return True

        self.prev_obs = mb_obs[end_len:]
        self.prev_rewards = mb_rewards[end_len:]
        self.prev_actions = mb_actions[end_len:]
        self.prev_values = mb_values[end_len:]
        self.prev_dones = mb_dones[end_len:]
        self.prev_log_probs = mb_log_probs[end_len:]
        mb_obs = mb_obs[:end_len]
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)[:end_len]
        mb_actions = mb_actions[:end_len]
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_log_probs = np.asarray(mb_log_probs, dtype=np.float32)[:end_len]
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        last_gae_lam = 0
        for t in reversed(range(end_len)):
            nextnonterminal = 1.0 - mb_dones[t]
            nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_values = mb_values[:end_len]
        mb_dones = mb_dones[:end_len]
        mb_returns = mb_advs + mb_values
        for i in range(len(mb_dones[0])):
            mb_dones[-1][i] = True
        mb_dones, mb_returns, mb_values, mb_log_probs = map(swap_and_flatten, (mb_dones, mb_returns, mb_values, mb_log_probs))
        mb_obs = [x for obs in mb_obs for x in obs]
        mb_actions = [x for obs in mb_actions for x in obs]
        return mb_obs, mb_dones, mb_returns, mb_actions, mb_values, mb_log_probs