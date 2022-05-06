import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .layers import Conv1d
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


class Critic(nn.Module):
    
    def __init__(self, hidden_node_dim):
        super().__init__()
        self.fc1 = Conv1d(hidden_node_dim, hidden_node_dim//2, init_scale=0.01)
        self.fc2 = Conv1d(hidden_node_dim//2, 1, init_scale=0.01)

    def forward(self, embs):
        output = F.relu(self.fc1(embs.transpose(2, 1)))
        value = self.fc2(output).sum(-1).squeeze()
        return value


class Policy(nn.Module):

    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim):
        super().__init__()
        self.encoder = Encoder(input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim)
        self.decoder = Decoder(hidden_node_dim, hidden_node_dim)
        self.value_net = Critic(hidden_node_dim)
        self.disc = nn.Sequential(
            nn.Linear(hidden_node_dim, hidden_node_dim),
            nn.Tanh(),
            nn.Linear(hidden_node_dim, 1)
        )

    def process_encoder(self, data):
        full_data = DataLoader(data, batch_size=len(data))
        for item in full_data:
            embs = self.encoder(item.to(self.device))
        solutions = [x.solution for x in data]
        mask_ids = [x.mask_ids for x in data]
        max_len = max([len(x) for x in solutions])
        input_embs, input_pool = [], []
        masks = torch.zeros((len(embs), max_len), dtype=torch.bool, device=self.device)
        for i, emb in enumerate(embs):
            pool = torch.mean(emb, 0)
            emb = emb[solutions[i]]
            masks[i][mask_ids[i]] = True
            input_emb = F.pad(emb, (0, 0, 0, max_len - len(emb)))
            input_embs.append(input_emb)
            input_pool.append(pool)
        input_embs = torch.stack(input_embs)
        input_pool = torch.stack(input_pool)
        return input_embs, input_pool, masks

    def forward(self, data, actions=None):
        input_embs, input_pool, masks = self.process_encoder(data)
        values = self.value_net(input_embs)
        if actions is None:
            actions, log_probs = self.decoder(input_embs, input_pool, masks)
            return actions, values, log_probs
        else:
            entropy, log_probs = self.decoder(input_embs, input_pool, masks, actions)
            return entropy, values, log_probs, input_pool
    
    def normalize(self, x):
        x = x.float()
        x = F.normalize(x, -1)
        return x

    def forward_loss(self, obs, dones, actions, old_values, returns, old_log_probs, advs, clip_range, use_imitation=True):
        entropy, vpred, log_probs, embs = self.forward(obs, actions)
        vpred_clipped = old_values + torch.clamp(vpred-old_values, -clip_range, clip_range)

        vf_losses1 = (vpred - returns)**2
        vf_losses2 = (vpred_clipped - returns)**2
        vf_loss = 0.5*torch.max(vf_losses1, vf_losses2).mean()

        ratio = torch.exp(log_probs - old_log_probs)
        pg_losses = advs * ratio
        pg_losses2 = advs * torch.clamp(ratio, 1.0-clip_range, 1.0+clip_range)
        pg_loss = -torch.min(pg_losses, pg_losses2).mean()

        imitation_loss = torch.tensor(0.0).to(entropy.device)
        if use_imitation:
            expert_ids = []
            student_ids = []
            for id, done in enumerate(dones):
                if done or id == len(dones) - 1:
                    expert_ids.append(id)
                else:
                    student_ids.append(id)
            expert_embs = embs[expert_ids]
            student_embs = embs[student_ids]
            logits_pi = self.disc(student_embs)
            logits_exp = self.disc(expert_embs)
            loss_pi = -F.logsigmoid(-logits_pi).mean()
            loss_exp = -F.logsigmoid(logits_exp).mean()
            with torch.no_grad():
                target_ids = torch.zeros(len(dones), dtype=torch.long)
                for i in range(len(dones)):
                    id = len(dones)-1-i
                    if dones[id] or i == 0:
                        target_id = id
                    target_ids[id] = target_id
                target_embs = embs[target_ids]
            loss_delta = F.mse_loss(embs, target_embs)
            imitation_loss = loss_pi + loss_exp + loss_delta 
        
        return entropy, vf_loss, pg_loss, imitation_loss
