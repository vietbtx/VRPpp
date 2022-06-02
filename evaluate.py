import os
import torch
from env import VectorizedVRP
import numpy as np
from runner import Runner
from torch.utils.data import DataLoader
from network.model import PPO, Policy
import argparse
from env import utils

torch.multiprocessing.set_sharing_strategy('file_system')

def run():
    step = 0
    while True:
        results = runner.run()
        if results is None:
            print("Exiting...")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])
    parser.add_argument('--n-envs', type=int, default=1)
    parser.add_argument('--n-steps', type=int, default=16)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--egde-dim', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--accumulation-iteration', type=int, default=1)
    parser.add_argument('--data-folder', type=str, default='dataset/data_cvrp')
    parser.add_argument('--min-extend-nodes', type=int, default=256)
    parser.add_argument('--min-extend-tours', type=int, default=4)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--clip-range', type=float, default=0.2)
    parser.add_argument('--imitation-rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--max-steps', type=int, default=64)
    parser.add_argument('--max-count', type=int, default=200)
    parser.add_argument('--reward-norm', type=int, default=100)
    parser.add_argument('--round-int', action='store_true')
    parser.add_argument('--algo', type=str, default='HGS', choices=['VNS', 'HGS'])
    parser.add_argument('--model-path', type=str)

    args = parser.parse_args()
    print("Running:", args)
    utils.set_seed(args.seed)
    env = VectorizedVRP(args)
    policy = Policy(3, args.hidden_dim, 2, args.egde_dim)
    ppo = PPO(policy, args)
    policy.load_state_dict(torch.load(args.model_path))
    policy.eval()
    name = "_".join(os.path.normpath(args.data_folder).split(os.sep))
    log_name = f"{name}_{args.algo}_{args.imitation_rate}/seed_{args.seed}"
    runner = Runner(env, policy, log_name, args)
    run()
    


    