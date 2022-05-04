import torch
from .policy import Policy
try:
    from apex import amp
except:
    amp = None


class PPO:

    def __init__(self, policy:Policy, args):
        self.policy = policy
        self.device = args.device
        self.ent_coef = args.ent_coef
        self.vf_coef = args.vf_coef
        self.max_grad_norm = args.max_grad_norm
        self.clip_range = args.clip_range
        self.imitation_rate = args.imitation_rate

        self.policy.to(args.device)
        self.policy.device = args.device
        self.policy.encoder.device = args.device
        self.policy.decoder.device = args.device

        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=args.lr, eps=args.eps)
        if amp is not None and self.device != 'cpu':
            self.policy, self.optimizer = amp.initialize(self.policy, self.optimizer, opt_level='O1', verbosity=0)

    def train(self, obs, dones, returns, actions, old_values, old_log_probs):
        old_values = torch.FloatTensor(old_values).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        advs = returns - old_values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        
        max_len = max([len(x) for x in actions])
        actions = [x + [0] * (max_len - len(x)) for x in actions]
        actions = torch.LongTensor(actions).to(self.device).T

        use_imitation = self.imitation_rate > 0
        entropy, vf_loss, pg_loss, imitation_loss = self.policy.forward_loss(obs, dones, actions, old_values, returns, old_log_probs, advs, self.clip_range, use_imitation)
        loss = pg_loss + self.vf_coef*vf_loss - self.ent_coef*entropy
        if use_imitation:
            loss += self.imitation_rate * imitation_loss
        
        if amp is not None and self.device != 'cpu':
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        return pg_loss.item(), vf_loss.item(), entropy.item(), imitation_loss.item()
