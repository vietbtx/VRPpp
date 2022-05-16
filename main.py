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


def train(obs, dones, returns, actions, values, log_probs, batch_size=128, n_epochs=1, accum_iter=1):
    lossvals = []
    for _ in range(n_epochs):
        dataloader = DataLoader(list(range(len(obs))), batch_size=batch_size, shuffle=False)  # don't use shuffle
        for batch_idx, ids in enumerate(dataloader):
            torch.cuda.empty_cache()
            obs_batch = [obs[i] for i in ids]
            actions_batch = [actions[i] for i in ids]
            losses = ppo.train(obs_batch, dones[ids], returns[ids], actions_batch, values[ids], log_probs[ids])
            lossvals.append(losses)
            if (batch_idx + 1) % accum_iter == 0 or batch_idx + 1 == len(dataloader):
                ppo.optimizer.step()
                ppo.optimizer.zero_grad()
    torch.cuda.empty_cache()
    lossvals = np.mean(lossvals, 0)
    return lossvals


def run(batch_size=128, n_epochs=1, accum_iter=1):
    step = 0
    while True:
        results = runner.run()
        if results is None:
            print("Exiting...")
            break
        obs, dones, returns, actions, values, log_probs = results
        lossvals = train(obs, dones, returns, actions, values, log_probs, batch_size, n_epochs, accum_iter)
        runner.writter.add_scalar(f"Losses/pg_loss", lossvals[0], step)
        runner.writter.add_scalar(f"Losses/vf_loss", lossvals[1], step)
        runner.writter.add_scalar(f"Losses/entropy", lossvals[2], step)
        runner.writter.add_scalar(f"Losses/imitation", lossvals[3], step)
        lossvals = [f"{loss:.3f}" for loss in lossvals]
        lossvals = ", ".join(lossvals)
        solved_dict = {k: max(v) for k, v in runner.solved_rate_dict.items()}
        keys = sorted(solved_dict.keys(), key=lambda x: solved_dict[x], reverse=True)
        processed = {k.split(".")[0]: f"{solved_dict[k]*100:.0f}%" for k in keys[:4]}
        processed = ", ".join(f"{k}: {v}" for k, v in processed.items())
        print(f"Step: {step} - loss: [{lossvals}] - {processed}")
        step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--n-envs', type=int, default=32)
    parser.add_argument('--n-steps', type=int, default=16)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--egde-dim', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--n-epochs', type=int, default=1)
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
    args = parser.parse_args()
    print("Running:", args)
    utils.set_seed(args.seed)
    env = VectorizedVRP(args)
    policy = Policy(3, args.hidden_dim, 2, args.egde_dim)
    ppo = PPO(policy, args)
    print("policy:", policy)
    name = "_".join(os.path.normpath(args.data_folder).split(os.sep))
    log_name = f"{name}_{args.algo}_{args.imitation_rate}/seed_{args.seed}"
    runner = Runner(env, policy, log_name, args)
    run(args.batch_size, args.n_epochs, args.accumulation_iteration)
    