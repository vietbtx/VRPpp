import torch
from env.VRPEnv import VRP, read_instances
from env.utils import set_seed, memory
from network.model import PPO, Policy
import argparse
import time
from torch.utils.tensorboard import SummaryWriter
from runner import load_scores, save_scores


def load_model(args, seed=1):
    policy = Policy(3, args.hidden_dim, 2, args.egde_dim)
    PPO(policy, args.device)
    log_name = f"seed_{seed}"
    if args.init_tour:
        log_name = f"{log_name}_init_tour"
    policy.load_state_dict(torch.load(f'logs/{log_name}/model.pt'))
    policy.eval()
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--n-envs', type=int, default=16)
    parser.add_argument('--n-steps', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--egde-dim', type=int, default=16)
    parser.add_argument('--init-tour', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--reward-norm', type=float, default=1.0)
    args = parser.parse_args()

    writter = SummaryWriter(f"logs/rand_instances")
    

    for seed in range(1, 15):
        print("Start:", f"data_evrp_random/seed_{seed}")
        policy = load_model(args, seed)
        set_seed(seed)
        memory.clear()
        best_scores = load_scores(f"logs/rand_instances/seed_{seed}")
        instances = read_instances(f"data_evrp_random/seed_{seed}", args.init_tour, seed)
        for instance in instances:
            env = VRP([instance], seed)
            count = 0
            step = 0
            best_score = None
            t0 = time.time()
            while count <= 20:
                print(f"s: {step+1} ", end="\r")
                step += 1
                state = env.state()
                with torch.no_grad():
                    actions, _, _ = policy.forward([state])
                    actions = actions.cpu().detach().tolist()
                _, _, score = env.step((actions[0], {}))
                if score is not None:
                    if best_score is None or score.score < best_score.score:
                        best_score = score
                        best_scores[score.name] = score
                        save_scores(best_scores, f"logs/rand_instances/seed_{seed}")
                    t = time.time() - t0
                    print("\nscore:", count, score, f"\t{t:.3f}s")
                    writter.add_scalar(f"seed_{seed}/{score.name}", score.score, count)
                    t0 = time.time()
                    count += 1
                    step = 0
                    
            

            

