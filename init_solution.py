import os
import torch
from env import init_instances
from runner import Runner
import argparse
from env import utils

torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-envs', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-folder', type=str, default='dataset/data_cvrp')
    parser.add_argument('--round-int', action='store_true')
    parser.add_argument('--algo', type=str, default='HGS', choices=['VNS', 'HGS'])
    args = parser.parse_args()
    print("Initializing:", args)
    init_instances(args)